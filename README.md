# 🎨 Neural Style Transfer

Apply the artistic style of famous paintings to any image using a pre-trained **VGG19** neural network built with **PyTorch**.

> 📌 Task-05 | Prodigy Infotech Internship

---

## 📸 Example

| Content Image | Style Image | Output |
|---|---|---|
| Your photo | Van Gogh – Starry Night | Stylized result |

---

## 🧠 How It Works

1. **VGG19** extracts deep features from both the content and style image
2. **Content Loss** — preserves the structure of your photo
3. **Style Loss** — captures texture and brushstrokes using Gram Matrices
4. **L-BFGS Optimizer** — iteratively updates pixels to minimize both losses

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Neural-Style-Transfer.git
cd Neural-Style-Transfer
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Your Images
Place your images inside the `images/` folder.

### 5. Run
```bash
python neural_style_transfer.py --content images/your_photo.jpg --style images/your_painting.jpg --output images/result.jpg --steps 200
```

---

## ⚙️ Arguments

| Argument | Description | Default |
|---|---|---|
| `--content` | Path to content image | required |
| `--style` | Path to style image | required |
| `--output` | Path to save result | `output.jpg` |
| `--steps` | Optimization steps (more = better) | `300` |
| `--style_weight` | Strength of style applied | `1000000` |
| `--content_weight` | Strength of content preserved | `1` |

---

## 🛠️ Tech Stack

- Python 3.8+
- PyTorch
- Torchvision
- Pillow

---

## 📚 References

- [A Neural Algorithm of Artistic Style – Gatys et al. (2015)](https://arxiv.org/abs/1508.06576)
- [PyTorch NST Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [VGG19 Paper](https://arxiv.org/abs/1409.1556)

---

## 👩‍💻 Author

**Vabhravi Pandey** — Prodigy Infotech Internship
