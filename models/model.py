"""Phase 2.4: model assembly.

Wraps target decoder layers of a frozen MiniCPM5-1B with a per-layer projector +
gated cross-attention block. Image features are stashed on the model per forward
(not threaded through the HF signature). Everything except projectors + xattn
(+ optional resampler) is frozen.
"""
import torch
import torch.nn as nn

from .projector import Projector, PerceiverResampler
from .xattn import GatedCrossAttn


class LayerWithXAttn(nn.Module):
    """Holds the original frozen decoder layer + its projector + xattn block.

    Runs the original layer, then applies gated cross-attention to its hidden state.
    Reads image features stashed on the parent model (owner._img_features) and the
    blind flag (owner._blind).
    """
    def __init__(self, orig_layer, projector, xattn, owner, resampler=None):
        super().__init__()
        self.orig_layer = orig_layer
        self.projector = projector
        self.xattn = xattn
        self.resampler = resampler
        self._owner = [owner]  # list to avoid registering owner as a submodule

    def forward(self, hidden_states, *args, **kwargs):
        out = self.orig_layer(hidden_states, *args, **kwargs)
        # HF decoder layers return a tuple (hidden, [attn], [cache], ...)
        if isinstance(out, tuple):
            hidden = out[0]
            rest = out[1:]
        else:
            hidden = out
            rest = None

        owner = self._owner[0]
        img = getattr(owner, "_img_features", None)
        if img is not None and not getattr(owner, "_blind", False):
            feats = img
            if self.resampler is not None:
                feats = self.resampler(feats)
            feats = self.projector(feats.to(hidden.dtype))
            hidden = self.xattn(hidden, feats)

        if rest is not None:
            return (hidden,) + rest
        return hidden


class VQAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = cfg["model"]
        self.cfg = cfg
        self.insertion_layers = list(m["insertion_layers"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            m["llm_name"], trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            m["llm_name"], torch_dtype=torch.float16,
            attn_implementation="sdpa", trust_remote_code=True)

        # Locate decoder layers (MiniCPM declares LlamaForCausalLM => model.model.layers)
        self.decoder = self.llm.model
        layers = self.decoder.layers
        n_layers = len(layers)
        print(f"[model] LLM loaded: {n_layers} decoder layers")
        assert max(self.insertion_layers) < n_layers, (
            f"insertion layer {max(self.insertion_layers)} >= {n_layers}")

        # Freeze the whole LLM.
        self.llm.requires_grad_(False)

        D = m["llm_hidden"]
        Vh = m["vision_hidden"]

        self.resampler = None
        if m.get("use_resampler", False):
            self.resampler = PerceiverResampler(
                dim=Vh, num_latents=m["resampler_tokens"])

        # One projector + one xattn per insertion layer.
        self.projectors = nn.ModuleList()
        self.xattns = nn.ModuleList()
        self._img_features = None
        self._blind = False

        for li in self.insertion_layers:
            proj = Projector(in_dim=Vh, out_dim=D)
            xa = GatedCrossAttn(embed_dim=D, num_heads=m["xattn_heads"])
            self.projectors.append(proj)
            self.xattns.append(xa)
            layers[li] = LayerWithXAttn(
                layers[li], proj, xa, owner=self, resampler=self.resampler)

        self.trainable_parameters(verbose=True)

    def set_image_features(self, img):
        self._img_features = img

    def clear_image_features(self):
        self._img_features = None

    def forward(self, input_ids, attention_mask, image_features, labels=None,
                blind=False):
        self._blind = blind
        self.set_image_features(None if blind else image_features)
        try:
            out = self.llm(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels)
        finally:
            self.clear_image_features()
            self._blind = False
        return out

    def trainable_parameters(self, verbose=False):
        total = 0
        groups = {"projectors": 0, "xattns": 0, "resampler": 0}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            total += p.numel()
            if n.startswith("projectors"):
                groups["projectors"] += p.numel()
            elif n.startswith("xattns"):
                groups["xattns"] += p.numel()
            elif n.startswith("resampler"):
                groups["resampler"] += p.numel()
        if verbose:
            print(f"[model] trainable params: {total:,}")
            for k, v in groups.items():
                if v:
                    print(f"[model]   {k}: {v:,}")
        return total

    def trainable_state_dict(self):
        """Only trainable tensors (projectors + xattn + gates + resampler)."""
        return {n: p.detach().cpu() for n, p in self.named_parameters()
                if p.requires_grad}

    def load_trainable_state_dict(self, sd):
        own = dict(self.named_parameters())
        missing = [k for k in sd if k not in own]
        assert not missing, f"unexpected keys: {missing[:5]}"
        with torch.no_grad():
            for k, v in sd.items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))

    def gate_report(self):
        """{layer_idx: (tanh_alpha, tanh_beta)} for logging / the gate figure."""
        return {li: xa.gate_magnitudes()
                for li, xa in zip(self.insertion_layers, self.xattns)}
