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


console = Console()
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path("../").resolve()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "coco_full_data.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
console.print(f"[bold green]Using device:[/bold green] {device}")

# ----------------------------
# Load dataset
# ----------------------------
full_df = pd.read_csv(DATA_PATH)

assert "image_id" in full_df.columns, \
    "coco_full_data.csv must contain image_id column."

# ----------------------------
# Sample 20k unique images
# ----------------------------
np.random.seed(42)
unique_images = full_df["image_id"].unique()

selected_images = np.random.choice(
    unique_images,
    size=20000,
    replace=False
)

subset_df = full_df[full_df["image_id"].isin(selected_images)]

console.print(f"[bold cyan]Total subset samples:[/bold cyan] {len(subset_df)}")

# ----------------------------
# Split 20k subset into train/val
# ----------------------------
train_imgs, val_imgs = train_test_split(
    selected_images,
    test_size=0.2,   # 80% train, 20% val
    random_state=42
)

train_df = subset_df[subset_df["image_id"].isin(train_imgs)]
val_df   = subset_df[subset_df["image_id"].isin(val_imgs)]

console.print(f"[bold cyan]Train samples:[/bold cyan] {len(train_df)}")
console.print(f"[bold cyan]Val samples:[/bold cyan] {len(val_df)}")

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "coco_train_20k.csv"
VAL_CSV   = PROJECT_ROOT / "data" / "processed" / "coco_val_20k.csv"

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)

# ----------------------------
# Load model
# ----------------------------
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ----------------------------
# Freeze entire backbone
# ----------------------------
for param in model.parameters():
    param.requires_grad = False

# ----------------------------
# Apply LoRA to text encoder
# ----------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model.text_model = get_peft_model(model.text_model, peft_config)

console.print("[bold yellow]LoRA applied to text encoder.[/bold yellow]")

# ----------------------------
# Build datasets
# ----------------------------
train_dataset = CocoCLIPDataset(TRAIN_CSV)
val_dataset   = CocoCLIPDataset(VAL_CSV)

trainer = CLIPTrainer(
    model=model,
    processor=processor,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    batch_size=32,      # LoRA allows bigger batch
    lr=5e-5,            # Higher LR ok for LoRA
    patience=3
)

history = trainer.train(
    epochs=20,
    checkpoint_dir=CHECKPOINT_DIR
)

# ----------------------------
# Save metrics
# ----------------------------
metrics_path = CHECKPOINT_DIR / "training_metrics.csv"
pd.DataFrame(history).to_csv(metrics_path, index=False)

console.print(f"[bold green]Metrics saved to {metrics_path}[/bold green]")