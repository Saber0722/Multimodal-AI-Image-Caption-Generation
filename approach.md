# 🖼️ Multimodal AI: Image Captioning with Vision Encoder + Transformer Decoder

## 📌 Project Overview

This project builds a **multimodal encoder–decoder architecture** that generates natural language captions from images.

The system consists of:

* **Vision Encoder (CNN or ViT)** → Extracts image features
* **Transformer Decoder** → Generates text captions autoregressively
* **Evaluation Pipeline** → BLEU, ROUGE metrics
* **Decoding Comparison** → Greedy vs Beam Search

This is a modern adaptation of the classic *Show and Tell* model, upgraded with Transformer-based decoding.

---

# 🧠 High-Level Architecture

```
Image → Vision Encoder → Feature Embeddings → Transformer Decoder → Caption
```

Detailed flow:

```
Input Image
      ↓
Pretrained CNN / ViT
      ↓
Image Feature Embeddings
      ↓
Projection Layer (to decoder dimension)
      ↓
Transformer Decoder (masked self-attention + cross-attention)
      ↓
Linear + Softmax
      ↓
Generated Caption
```

---

# 🏗 System Architecture Components

---

## 1️⃣ Dataset


### Option A: COCO Mini

![Image](https://www.researchgate.net/publication/376235886/figure/fig6/AS%3A11431281241523148%401715134617560/Sample-images-from-the-COCO-dataset-and-captions-predicted-by-the-Base-version-of-our.png)

![Image](https://cocodataset.org/images/coco-examples.jpg)

![Image](https://www.labellerr.com/blog/content/images/2023/06/Screenshot-2023-05-31-234904.png)

![Image](https://cdn.labellerr.com/COCO/Screenshot%202023-06-01%20100603.webp)

* Subset of MS COCO
* Rich object diversity
* Better generalization

### Option B: Coco Large
---

## 2️⃣ Vision Encoder (Image Feature Extractor)

Two strong choices:

---

Below is the **modified documentation section**, aligned with your final three experimental setups:

---

# 🔹 Option A: ResNet-50 Encoder (COCO-mini) + GPT Decoder

![Image](https://www.researchgate.net/publication/356162462/figure/fig3/AS%3A1089285335846971%401636717270496/The-architecture-of-the-ResNet-50-network.jpg)

![Image](https://www.researchgate.net/publication/350524328/figure/fig1/AS%3A1007436949364737%401617203094867/Resnet-Architectures-Right-And-Residual-Block-Top-Left-Bottleneck-Layer-Bottom.ppm)

![Image](https://www.researchgate.net/publication/376497473/figure/fig2/AS%3A11431281212236571%401702573294749/An-example-for-feature-map-visualizations-in-ResNet50-The-input-images-contain-NIR-Red.png)

![Image](https://www.researchgate.net/publication/337538111/figure/fig5/AS%3A901560490012674%401591960179448/sualization-of-the-feature-maps-in-the-ResNet-50-and-VGG-16-models-trained-by-the-two.jpg)

### 📌 Configuration

* **Encoder:** Microsoft Research ResNet-50
* **Dataset:** COCO-mini
* **Decoder:** GPT-style Transformer decoder

---

### 🧠 Architecture

**Encoder Modifications**

* Remove final classification layer
* Extract:

  * Global pooled feature (2048-d), or
  * Spatial feature map (7×7×2048)

**Recommended for captioning:**

* Use spatial features for cross-attention

---

### 📦 Output Formats

| Type             | Shape         | Usage                         |
| ---------------- | ------------- | ----------------------------- |
| Global vector    | (B, 2048)     | Simple prefix conditioning    |
| Spatial features | (B, 49, 2048) | Cross-attention (recommended) |

---

### 🎯 Why This Setup?

✔ Stable baseline
✔ Lightweight (fits 8GB easily)
✔ Works well for medium-small dataset
✔ Clean encoder–decoder separation

This serves as your **controlled baseline experiment**.

---

# 🔹 Option B: CLIP Vision Encoder (COCO-Large ~80k Images)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2Ah5xJzfFAfjdysNvqQbB9nQ.png)

![Image](https://miro.medium.com/1%2AuVUI6bU49oT-nNRGFs4GEA.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2Al37va2Mu8Snx6LLb13430A.png)

![Image](https://substackcdn.com/image/fetch/%24s_%21R7V_%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F998aad2b-fb98-458c-9cbb-4ba036e32e60_800x565.png)

### 📌 Configuration

* **Encoder:** CLIP Vision Encoder (ViT-B/16)
* Developed by OpenAI
* **Dataset:** COCO Large (~80k images)
* **Decoder:** GPT-style decoder

---

### 🧠 How CLIP Helps

CLIP is pretrained on **400M image–text pairs** using contrastive learning.

Instead of classification, it learns:

> Image ↔ Text semantic alignment

This makes CLIP embeddings:

* Language-aware
* Context-aware
* Better suited for caption generation

---

### 📦 Output

* Patch embeddings: (B, N, 768)
* Already aligned to text semantic space

---

### 🎯 Why This Setup?

✔ Strongest semantic understanding
✔ Best expected BLEU/ROUGE
✔ Ideal for large dataset (80k images)
✔ Reduces learning burden on decoder

This is your **high-performance configuration**.

---

# 🔹 Option C: Vision Transformer (ViT-B/16) + GPT Decoder (COCO-Large)

![Image](https://blog.roboflow.com/content/images/2025/04/Screenshot-2025-04-17-at-1.30.34-PM.png)

![Image](https://cdn.sanity.io/images/vr8gru94/production/7a096efc8f3cc40849ee17a546dc0e685da2dc73-4237x1515.png)

![Image](https://theaisummer.com/static/aa65d942973255da238052d8cdfa4fcd/7d4ec/the-transformer-block-vit.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2Al37va2Mu8Snx6LLb13430A.png)

### 📌 Configuration

* **Encoder:** ViT-B/16
* Developed by Google Research
* **Dataset:** COCO Large (~80k images)
* **Decoder:** GPT-style Transformer decoder

---

### 🧠 How It Works

* Split image into 16×16 patches
* Linear projection → patch embeddings
* Add positional encodings
* Process through transformer encoder

Output:

* Patch embeddings (B, N, 768)

---

### 🎯 Why This Setup?

✔ Pure transformer pipeline
✔ Global self-attention
✔ Cleaner theoretical alignment with GPT decoder
✔ Good scaling with larger datasets

This serves as your **pure transformer experiment**.

---

# 3️⃣ Feature Projection Layer (All Setups)

Since encoder output dimension ≠ decoder dimension:

```
Image Features → Linear Layer → d_model (e.g., 512 or 768)
```

### Purpose:

* Align encoder embeddings with GPT decoder input dimension
* Enable cross-attention compatibility

Example:

* ResNet output: 2048 → 512
* ViT/CLIP output: 768 → 512

---

# 4️⃣ GPT-Style Transformer Decoder (Text Generator)

![Image](https://miro.medium.com/0%2A376uJu_fc_uR8H3X.png)

![Image](https://cdn.sanity.io/images/jo7n4k8s/production/25ebbba9d2ce12efc8c3da181942367f05c795be-2386x1338.jpg?auto=format)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AxzvpKDgLm2A-D9C04V4rOw.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A2000/1%2AF4EZBBYoQN3pAqk8tXC8sA.png)

---

## 🧠 Decoder Architecture

Each decoder layer contains:

### 1️⃣ Masked Self-Attention

* Prevents access to future tokens
* Autoregressive generation

### 2️⃣ Cross-Attention

* Queries = text tokens
* Keys/Values = image embeddings

### 3️⃣ Feed Forward Network

* Position-wise MLP
* Non-linear transformation

---

## 📥 Input to Decoder

* `<SOS>` token
* Previously generated tokens
* Positional encoding

---

## 📤 Output Layer

```
Decoder Output → Linear Layer → Vocabulary Size → Softmax
```

Produces probability distribution over next word.

---

# 🔬 Summary of Your Three Experiments

| Setup           | Dataset    | Strength                | Purpose                 |
| --------------- | ---------- | ----------------------- | ----------------------- |
| ResNet-50 + GPT | COCO-mini  | Stable baseline         | Controlled experiment   |
| CLIP + GPT      | COCO-large | Best semantic alignment | Highest performance     |
| ViT + GPT       | COCO-large | Pure transformer        | Architecture comparison |

---

If you'd like, I can now:

* Add expected BLEU/ROUGE per setup
* Provide architectural diagrams combining encoder + GPT
* Add training strategy per configuration
* Help you write the experimental comparison section for your report


# 🔁 Training Pipeline

### 1️⃣ Preprocessing

* Resize image (224×224)
* Normalize (ImageNet stats)
* Tokenize captions
* Pad sequences
* Add `<SOS>`, `<EOS>`

---

### 2️⃣ Training Objective

**Cross-Entropy Loss**

```
Loss = - Σ log P(target_word | previous_words, image)
```

Teacher forcing used during training.

---

### 3️⃣ Optimization

* Adam optimizer
* Learning rate scheduler (optional)
* Freeze encoder initially (recommended for small datasets)

---

# 🔎 Inference & Decoding Strategies

---

## 🔹 Greedy Decoding

```
At each step:
    Select word with highest probability
```

✔ Fast
✖ Can miss better global sequence

---

## 🔹 Beam Search

```
Keep top-k candidate sequences at each step
```

Example:

Beam size = 3
Keep 3 best partial sentences at every timestep.

✔ Better captions
✖ Slower

---

### Expected Comparison

| Metric | Greedy   | Beam Search |
| ------ | -------- | ----------- |
| BLEU   | Moderate | Higher      |
| ROUGE  | Moderate | Higher      |
| Speed  | Fast     | Slower      |

---

# 📊 Evaluation Metrics

---

## 🔹 BLEU (Bilingual Evaluation Understudy)

Measures n-gram precision overlap.

BLEU-1 → Unigram
BLEU-4 → 4-gram

Higher = better

---

## 🔹 ROUGE

Measures recall overlap.

ROUGE-L → Longest Common Subsequence

---

## Optional: CIDEr (if dataset supports)

Better for image captioning tasks.

---

# 📦 Full Model Architecture Diagram (Conceptual)

```
                ┌──────────────────────┐
                │      Input Image     │
                └─────────┬────────────┘
                          ↓
              ┌──────────────────────┐
              │   CNN / ViT Encoder  │
              └─────────┬────────────┘
                        ↓
              ┌──────────────────────┐
              │   Feature Projection │
              └─────────┬────────────┘
                        ↓
         ┌─────────────────────────────────┐
         │        Transformer Decoder      │
         │  Masked Self-Attention          │
         │  Cross Attention (Image)        │
         │  Feed Forward                   │
         └─────────┬───────────────────────┘
                   ↓
          ┌──────────────────────┐
          │  Linear + Softmax    │
          └─────────┬────────────┘
                    ↓
               Generated Caption
```

---
