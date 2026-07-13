
# Loads the NegCLIP (or vanilla CLIP) vision+text encoder and the image pre-processing transform.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import open_clip
import torch

# Re-export the tokenizer so callers only need to import from this module.
from open_clip import get_tokenizer  # noqa: F401

# Import config relative to the project root.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    #Return the best available device (CUDA > MPS > CPU).
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, object, object]:

    #Load and return (model, preprocess_transform, tokenizer).
    if device is None:
        device = get_device()

    logger.info("Loading model arch=%s on device=%s", config.MODEL_ARCH, device)

    # Always create the architecture with the OpenCLIP factory so that
    # pre-processing is consistent regardless of weight source.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=config.MODEL_ARCH,
        pretrained=config.PRETRAINED_TAG,
        device=device,
    )

    # --- Override weights with local NegCLIP checkpoint if provided ----------
    negclip_path = config.NEGCLIP_WEIGHTS_PATH
    if negclip_path is not None:
        negclip_path = Path(negclip_path)
        if not negclip_path.is_file():
            raise FileNotFoundError(
                f"NEGCLIP_WEIGHTS_PATH is set but the file was not found: "
                f"{negclip_path}"
            )
        logger.info("Loading NegCLIP weights from %s", negclip_path)
        state_dict = torch.load(negclip_path, map_location=device)

        # Checkpoints may be wrapped in a "state_dict" key.
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Strip "module." prefix from DataParallel-saved checkpoints.
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("NegCLIP checkpoint: missing keys: %s", missing)
        if unexpected:
            logger.warning("NegCLIP checkpoint: unexpected keys: %s", unexpected)
        logger.info("NegCLIP weights loaded successfully.")
    else:
        logger.info(
            "No local NegCLIP weights found; using pretrained='%s'. "
            "Set NEGCLIP_WEIGHTS_PATH env var to use NegCLIP weights.",
            config.PRETRAINED_TAG,
        )

    model.eval()

    tokenizer = get_tokenizer(config.MODEL_ARCH)

    return model, preprocess, tokenizer
