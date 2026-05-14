# 🎨 Neural Style Transfer — Task 05
**Prodigy Infotech Internship Project**

Apply the artistic style of one image (e.g., Van Gogh's *Starry Night*) to the content of another image using a pre-trained **VGG19** neural network.

---

## 📁 Project Structure

```
neural_style_transfer/
├── neural_style_transfer.py   ← Core script (main logic)
├── demo.py                    ← Quick demo (auto-downloads sample images)
├── requirements.txt           ← Python dependencies
├── images/                    ← Put your content & style images here
│   ├── content.jpg
│   ├── style_starry_night.jpg
│   └── output.jpg             ← Generated output
└── .vscode/
    └── launch.json            ← VS Code run configurations
```

---

## ⚙️ Setup in VS Code

### Step 1 — Open the Project
```
File → Open Folder → select neural_style_transfer/
```

### Step 2 — Create a Virtual Environment (Recommended)
Open the **VS Code Terminal** (`Ctrl + `` ` ``) and run:
```bash
# Create venv
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ PyTorch installation may vary. Visit https://pytorch.org/get-started/locally/ if you face issues.

---

## 🚀 How to Run

### Option A — Quick Demo (no images needed)
```bash
python demo.py
```
This downloads sample images automatically and runs the transfer.

---

### Option B — Use Your Own Images
```bash
python neural_style_transfer.py \
    --content images/your_photo.jpg \
    --style   images/your_painting.jpg \
    --output  images/result.jpg \
    --steps   300
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--content` | Path to your content image | *(required)* |
| `--style` | Path to your style image | *(required)* |
| `--output` | Where to save the result | `output.jpg` |
| `--steps` | Optimization iterations (more = better quality, slower) | `300` |
| `--style_weight` | How strongly style is applied | `1000000` |
| `--content_weight` | How much content is preserved | `1` |

---

### Option C — Run from VS Code UI
1. Press **F5** or click **Run → Start Debugging**
2. Choose **"Run Demo"** or **"Custom NST (Edit Args)"**

---

## 🧠 How It Works

1. **VGG19 Pre-trained Network** extracts features from both the content and style image
2. **Content Loss** — Measures how similar the generated image is to the content image at a deep layer (`conv_4`)
3. **Style Loss** — Uses **Gram Matrices** at multiple layers (`conv_1` to `conv_5`) to capture texture/style
4. **L-BFGS Optimizer** iteratively updates the pixels of the output image to minimize both losses

```
Content Image ──┐
                ├──► VGG19 ──► Losses ──► L-BFGS ──► Output Image
Style Image   ──┘
```

---

## 🖼️ Example Results

| Content | Style | Output |
|---|---|---|
| Your photo | Van Gogh – Starry Night | Stylized photo |
| Your photo | Picasso – Guernica | Cubist version |
| Your photo | Monet – Water Lilies | Impressionist version |

---

## 💡 Tips

- **Faster results**: Use `--steps 100` for a quick preview
- **Better quality**: Use `--steps 500` or more
- **GPU**: If you have an NVIDIA GPU, install CUDA PyTorch for 10x speedup
- **Style intensity**: Increase `--style_weight` (e.g., `5000000`) for more stylization
- **Content preservation**: Increase `--content_weight` (e.g., `10`) to keep more of the original photo

---

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- Pillow

---

## 📚 References

1. Gatys et al. (2015) — *A Neural Algorithm of Artistic Style* — https://arxiv.org/abs/1508.06576
2. PyTorch NST Tutorial — https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
3. VGG19 Paper — https://arxiv.org/abs/1409.1556
