from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CocoCLIPDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        assert self.csv_path.exists(), "CSV file not found."

        self.df = pd.read_csv(self.csv_path)
        assert "image_path" in self.df.columns
        assert "caption" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        caption = row["caption"]

        assert image_path.exists(), f"Image not found: {image_path}"

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "caption": caption
        }
