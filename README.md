# 🧠 Brain Tumor Classification using Swin Transformer + CNN + BiLSTM

This project presents a **hybrid deep learning architecture** for automated **brain tumor classification from MRI images**.

The model combines:

- **Swin Transformer** for global feature extraction  
- **CNN branch** for local spatial feature extraction  
- **BiLSTM** for sequential feature learning  

The hybrid architecture improves representation learning and achieves **very high classification performance** on MRI brain tumor datasets.

---

# 📌 Problem Statement

Brain tumor detection using MRI scans is an important task in medical imaging.

Manual diagnosis by radiologists can be:

- Time consuming  
- Subjective  
- Error prone  

Therefore, automated AI systems can assist doctors by providing **accurate and fast tumor classification**.

The goal is to classify MRI images into **four categories**:

- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

---

# 🏗 Model Architecture

The proposed architecture integrates **three deep learning components**.

## 1️⃣ Swin Transformer

Pretrained model:

```
microsoft/swin-tiny-patch4-window7-224
```

Purpose:

- Extract **global contextual features**
- Capture long-range spatial dependencies in MRI images

Output feature size:

```
768
```

---

## 2️⃣ CNN Branch

A lightweight convolutional network extracts **local spatial features**.

Architecture:

```
Conv2D (3 → 32)
ReLU
MaxPooling

Conv2D (32 → 64)
ReLU
AdaptiveAvgPool
```

Output feature size:

```
64
```

---

## 3️⃣ Feature Fusion

Swin features and CNN features are combined.

```
Swin Feature (768)
+
CNN Feature (64)
----------------
Combined Feature = 832
```

These features are treated as a **sequence input to BiLSTM**.

---

## 4️⃣ BiLSTM Layer

The **Bidirectional LSTM** learns sequential dependencies from fused features.

Parameters:

```
Hidden Size = 128
Bidirectional = True
```

Output feature size:

```
256
```

---

## 5️⃣ Final Classifier

```
Linear (256 → num_classes)
```

Number of classes:

```
4
```

Classes:

- glioma
- meningioma
- no_tumor
- pituitary

---

# 🔬 Model Implementation

Main architecture:

```python
class SwinCNNBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.swin = SwinModel.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )

        self.cnn = CNNBranch()
        self.lstm = BiLSTMBranch(input_size=768 + 64, hidden_size=128)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        swin_out = self.swin(pixel_values=x).last_hidden_state
        cnn_vec = self.cnn(x)

        B, T, F = swin_out.shape
        cnn_rep = cnn_vec.unsqueeze(1).expand(B, T, cnn_vec.size(1))

        comb = torch.cat([swin_out, cnn_rep], dim=2)

        lstm_out = self.lstm(comb)

        logits = self.fc(lstm_out)

        return logits
```

---

# 📂 Dataset

MRI Brain Tumor Dataset with four classes:

```
glioma
meningioma
no_tumor
pituitary
```

Dataset structure:

```
dataset
│
├── train
│   ├── glioma
│   ├── meningioma
│   ├── no_tumor
│   └── pituitary
│
└── test
```

---

# ⚙ Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Swin Transformer + CNN + BiLSTM |
| Classes | 4 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Framework | PyTorch |

---

# 📊 Results

### Final Test Accuracy

```
0.9899
```

### Final Weighted F1 Score

```
0.9898
```

---

# 📉 Confusion Matrix

```
[[1958    0    0    0]
 [   0 1958    0    0]
 [  30   47 1879    2]
 [   0    0    0 1958]]
```

---

# 📑 Classification Report

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| Glioma | 0.98 | 1.00 | 0.99 |
| Meningioma | 0.98 | 1.00 | 0.99 |
| No Tumor | 1.00 | 0.96 | 0.98 |
| Pituitary | 1.00 | 1.00 | 1.00 |

Overall accuracy:

```
99%
```

---

# 🚀 Installation

Clone repository:

```
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

Install dependencies:

```
pip install torch
pip install torchvision
pip install transformers
pip install scikit-learn
pip install numpy
pip install matplotlib
```

---

# ▶ Running the Project

Train model:

```
python train.py
```

Evaluate model:

```
python test.py
```

---

# 📁 Project Structure

```
brain-tumor-classification
│
├── model.py
├── train.py
├── test.py
│
├── dataset
│
├── results
│   ├── confusion_matrix.png
│
└── README.md
```

---

# 🔮 Future Improvements

Possible extensions:

- Explainable AI (Grad-CAM / LIME)
- Federated Learning for medical privacy
- Larger MRI datasets
- Lightweight models for hospital deployment

---

# 👨‍💻 Author

**Md. Sohag Hossain**

CSE Student  
Machine Learning & AI Research Enthusiast

---

# ⭐ Acknowledgements

- PyTorch
- HuggingFace Transformers
- Swin Transformer Architecture
