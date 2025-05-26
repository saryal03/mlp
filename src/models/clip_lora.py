import os
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

# Optional PEFT imports for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    LoraConfig = None
    get_peft_model = None
    TaskType = None


def get_clip_with_lora(
    clip_model_name: str,
    local_model_path: str = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """
    Loads a CLIP model and processor, applying LoRA adapters if PEFT is available.

    Args:
      clip_model_name: Hugging Face repo ID for base CLIP (e.g. "openai/clip-vit-base-patch16").
      local_model_path: Optional local directory with pretrained model files.
      lora_rank: LoRA rank (r).
      lora_alpha: LoRA alpha scaling.
      lora_dropout: LoRA dropout probability.

    Returns:
      model: CLIPModel (with LoRA adapters if installed).
      processor: CLIPProcessor for preprocessing inputs.
    """
    # 1. Load base model & processor (local if available)
    if local_model_path and Path(local_model_path).is_dir():
        model = CLIPModel.from_pretrained(local_model_path, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(local_model_path, local_files_only=True)
    else:
        model = CLIPModel.from_pretrained(clip_model_name)
        processor = CLIPProcessor.from_pretrained(clip_model_name)

    # 2. Apply LoRA adapters if PEFT is installed
    if get_peft_model and LoraConfig and TaskType:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_config)
    else:
        print("⚠️ PEFT not installed; returning base CLIP without LoRA adapters.")

    return model, processor
