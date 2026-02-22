import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np


class CLIPTrainer:
    def __init__(
        self,
        model,
        processor,
        train_dataset,
        val_dataset,
        device,
        batch_size=8,
        lr=5e-6,
        patience=5
    ):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.patience = patience

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler("cuda")

        logging.info("Trainer initialized.")

    def collate_fn(self, batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]

        return self.processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        loop = tqdm(self.train_loader)

        for batch in loop:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast("cuda"):
                outputs = self.model(**batch, return_loss=True)
                loss = outputs.loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            loop.set_description(f"Train Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        image_embeds = []
        text_embeds = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                image_embeds.append(outputs.image_embeds.cpu())
                text_embeds.append(outputs.text_embeds.cpu())

        image_embeds = torch.cat(image_embeds)
        text_embeds = torch.cat(text_embeds)

        image_embeds /= image_embeds.norm(dim=1, keepdim=True)
        text_embeds /= text_embeds.norm(dim=1, keepdim=True)

        similarity = image_embeds @ text_embeds.T

        recall1 = self.recall_at_k(similarity, 1)
        return recall1

    @staticmethod
    def recall_at_k(similarity, k):
        correct = 0
        for i in range(len(similarity)):
            if i in similarity[i].topk(k).indices:
                correct += 1
        return correct / len(similarity)

    def train(self, epochs, checkpoint_dir):
        best_score = 0
        patience_counter = 0
        history = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss = self.train_one_epoch()
            val_recall = self.validate()

            history.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "val_recall@1": val_recall
            })

            torch.save(self.model.state_dict(), checkpoint_dir / "last_model.pt")

            if val_recall > best_score:
                best_score = val_recall
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pt")
                print("Best model updated.")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        return history
