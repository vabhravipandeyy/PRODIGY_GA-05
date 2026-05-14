"""
Neural Style Transfer
Task-05 | Prodigy Infotech Internship

Applies the artistic style of one image (e.g., a famous painting)
to the content of another image using a pre-trained VGG19 network.

Usage:
    python neural_style_transfer.py --content <path> --style <path> --output <path>
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import argparse
import os

# ─── Device Setup ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ─── Image Size ──────────────────────────────────────────────────────────────
IMG_SIZE = 512 if torch.cuda.is_available() else 256
print(f"[INFO] Image size: {IMG_SIZE}x{IMG_SIZE}")

# ─── Image Loader ────────────────────────────────────────────────────────────
loader = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def load_image(image_path):
    """Load an image and convert to a tensor."""
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)  # Add batch dimension
    return image.to(device, torch.float)

def tensor_to_image(tensor):
    """Convert a tensor back to a PIL Image."""
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    return image

# ─── Loss Modules ────────────────────────────────────────────────────────────
class ContentLoss(nn.Module):
    """Compute the content loss between the target and generated image features."""
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


def gram_matrix(tensor):
    """Compute Gram matrix for style representation."""
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    """Compute the style loss using Gram matrices."""
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    """Normalize input images with ImageNet mean and std."""
    def __init__(self):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)
        self.mean = mean
        self.std  = std

    def forward(self, img):
        return (img - self.mean) / self.std

# ─── VGG19 Model Builder ──────────────────────────────────────────────────────
CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def build_style_model(cnn, content_img, style_img):
    """
    Build the model by inserting Content/Style loss modules after
    specific VGG19 conv layers.
    """
    cnn = copy.deepcopy(cnn)
    normalization = Normalization().to(device)

    content_losses = []
    style_losses   = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in CONTENT_LAYERS:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f'content_loss_{i}', cl)
            content_losses.append(cl)

        if name in STYLE_LAYERS:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f'style_loss_{i}', sl)
            style_losses.append(sl)

    # Trim layers after the last loss module
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j + 1]

    return model, style_losses, content_losses

# ─── Optimizer ───────────────────────────────────────────────────────────────
def get_input_optimizer(input_img):
    """Use L-BFGS optimizer on the input image pixels."""
    optimizer = optim.LBFGS([input_img.requires_grad_(True)])
    return optimizer

# ─── Style Transfer ───────────────────────────────────────────────────────────
def run_style_transfer(content_img, style_img,
                       num_steps=300,
                       style_weight=1_000_000,
                       content_weight=1):
    """
    Run the neural style transfer optimization loop.
    Returns the stylized output image tensor.
    """
    print("[INFO] Loading VGG19...")
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    print("[INFO] Building style model...")
    model, style_losses, content_losses = build_style_model(cnn, content_img, style_img)

    # Start from content image
    input_img = content_img.clone()
    optimizer = get_input_optimizer(input_img)

    print(f"[INFO] Starting optimization for {num_steps} steps...")
    run = [0]

    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score   = sum(sl.loss for sl in style_losses)   * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"  Step {run[0]:>4}/{num_steps} | "
                      f"Style: {style_score.item():.2f} | "
                      f"Content: {content_score.item():.4f}")
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content", type=str, required=True,
                        help="Path to the content image")
    parser.add_argument("--style",   type=str, required=True,
                        help="Path to the style image")
    parser.add_argument("--output",  type=str, default="output.jpg",
                        help="Output file path (default: output.jpg)")
    parser.add_argument("--steps",   type=int, default=300,
                        help="Number of optimization steps (default: 300)")
    parser.add_argument("--style_weight",   type=float, default=1_000_000,
                        help="Weight for style loss (default: 1e6)")
    parser.add_argument("--content_weight", type=float, default=1,
                        help="Weight for content loss (default: 1)")
    args = parser.parse_args()

    # Validate input files
    for path, name in [(args.content, "Content"), (args.style, "Style")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} image not found: {path}")
            return

    print(f"\n{'='*50}")
    print("      NEURAL STYLE TRANSFER - Task 05")
    print(f"{'='*50}")
    print(f"  Content : {args.content}")
    print(f"  Style   : {args.style}")
    print(f"  Output  : {args.output}")
    print(f"  Steps   : {args.steps}")
    print(f"{'='*50}\n")

    content_img = load_image(args.content)
    style_img   = load_image(args.style)

    output_tensor = run_style_transfer(
        content_img, style_img,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight
    )

    result = tensor_to_image(output_tensor)
    result.save(args.output)
    print(f"\n[✓] Stylized image saved to: {args.output}")


if __name__ == "__main__":
    main()
