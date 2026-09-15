"""Phase 2.1: frozen SigLIP vision wrapper -> [B, 196, 768] patch tokens.

At train time features are read from the precomputed cache (data/cache_features.py);
this live-forward path is used only for caching and the no-cache flag.
"""
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, name="google/siglip-base-patch16-224", device="cuda",
                 dtype=torch.float16):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        # image processor only (avoids pulling SiglipTokenizer/sentencepiece)
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).vision_model
        self.model.to(device=device, dtype=dtype).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode_pil(self, images):
        """images: list of PIL.Image -> [B, 196, 768] patch tokens (last_hidden_state)."""
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        out = self.model(pixel_values=pixel_values)
        # SigLIP vision_model returns patch tokens (no CLS) in last_hidden_state.
        return out.last_hidden_state  # [B, 196, 768]
