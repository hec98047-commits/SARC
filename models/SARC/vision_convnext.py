import torch
import torch.nn as nn


class Fgclip2ConvNeXtVisionPipeline(nn.Module):
    """Compatibility stub for checkpoints that ship only the ViT backbone files.

    The local FG-CLIP checkpoint in this workspace uses the default ViT vision path,
    but `transformers` still tries to copy this relative module because it is imported
    by `modeling_fgclip2.py`. Keeping this stub file present avoids a hard
    `FileNotFoundError` during dynamic module loading.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.convnext = None
        raise NotImplementedError(
            "ConvNeXt vision pipeline is not packaged in this local checkpoint. "
            "This stub is only provided so the ViT-based FG-CLIP model can load. "
            "If you need a ConvNeXt-backed checkpoint, add the original "
            "`vision_convnext.py` implementation and matching weights."
        )

    def forward(self, pixel_values, attention_mask, spatial_shapes):
        raise NotImplementedError("ConvNeXt vision pipeline stub cannot run inference.")
