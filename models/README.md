# Model source files

This directory contains only the FG-CLIP and MG-FGCLIP Python model implementations and their non-weight configuration files used by PGCRE-FGCLIP.

The following artifacts are intentionally excluded from Git:

- model weights (`*.safetensors`, `*.pth`, `*.pt`, `*.ckpt`);
- tokenizer vocabulary/model files;
- downloaded Hugging Face or Torch Hub caches;
- generated caches such as `__pycache__/`.

Supply the required checkpoint and tokenizer assets locally before running an experiment. See the repository README for the expected layout and command-line arguments.
