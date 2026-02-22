from pathlib import Path
import logging
import torch
import pandas as pd
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from rich.console import Console
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model

from dataset import CocoCLIPDataset
from trainer import CLIPTrainer


# --------------------------------------------------
# Setup
# --------------------------------------------------

console = Console()
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path("../").resolve()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "coco_full_data.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_full"
CHECKPOINT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
console.print(f"[bold green]Using device:[/bold green] {device}")


# --------------------------------------------------
# Load Full Dataset (NO SUBSAMPLING)
# --------------------------------------------------

assert DATA_PATH.exists(), "coco_full_data.csv not found."
full_df = pd.read_csv(DATA_PATH)

assert "image_id" in full_df.columns, \
    "coco_full_data.csv must contain image_id column."

unique_images = full_df["image_id"].unique()

console.print(f"[bold cyan]Total unique images:[/bold cyan] {len(unique_images)}")
console.print(f"[bold cyan]Total samples:[/bold cyan] {len(full_df)}")


# --------------------------------------------------
# Leakage-Safe Split (by image_id)
# --------------------------------------------------

train_imgs, val_imgs = train_test_split(
    unique_images,
    test_size=0.1,   # 90% train, 10% val
    random_state=42
)

train_df = full_df[full_df["image_id"].isin(train_imgs)]
val_df   = full_df[full_df["image_id"].isin(val_imgs)]

console.print(f"[bold cyan]Train samples:[/bold cyan] {len(train_df)}")
console.print(f"[bold cyan]Val samples:[/bold cyan] {len(val_df)}")


TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "coco_train_full.csv"
VAL_CSV   = PROJECT_ROOT / "data" / "processed" / "coco_val_full.csv"

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


# --------------------------------------------------
# Freeze Entire Backbone
# --------------------------------------------------

for param in model.parameters():
    param.requires_grad = False


# --------------------------------------------------
# Apply LoRA to Text Encoder
# --------------------------------------------------

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model.text_model = get_peft_model(
    model.text_model,
    peft_config
)

console.print("[bold yellow]LoRA applied to text encoder.[/bold yellow]")


# --------------------------------------------------
# ALSO Train Projection Heads (Important)
# --------------------------------------------------

for param in model.visual_projection.parameters():
    param.requires_grad = True

for param in model.text_projection.parameters():
    param.requires_grad = True

console.print("[bold yellow]Projection heads unfrozen.[/bold yellow]")


# --------------------------------------------------
# Build Datasets
# --------------------------------------------------

train_dataset = CocoCLIPDataset(TRAIN_CSV)
val_dataset   = CocoCLIPDataset(VAL_CSV)


# --------------------------------------------------
# Trainer Configuration (Scaled Version)
# --------------------------------------------------

trainer = CLIPTrainer(
    model=model,
    processor=processor,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    batch_size=32,       # Safe for 8GB VRAM
    lr=2e-5,             # Lower LR for full dataset
    patience=3
)


# --------------------------------------------------
# Train
# --------------------------------------------------

history = trainer.train(
    epochs=10,           # Full dataset needs fewer epochs
    checkpoint_dir=CHECKPOINT_DIR
)


# --------------------------------------------------
# Save Metrics
# --------------------------------------------------

metrics_path = CHECKPOINT_DIR / "training_metrics_full.csv"
pd.DataFrame(history).to_csv(metrics_path, index=False)

console.print(f"[bold green]Training complete.[/bold green]")
console.print(f"[bold green]Metrics saved to {metrics_path}[/bold green]")