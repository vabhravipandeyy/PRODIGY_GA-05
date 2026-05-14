"""
demo.py — Quick demo for Neural Style Transfer
"""

import urllib.request
import os
from neural_style_transfer import load_image, tensor_to_image, run_style_transfer

def download(url, filename):
    if not os.path.exists(filename):
        print(f"[↓] Downloading {filename}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as r, open(filename, 'wb') as f:
            f.write(r.read())
        print(f"    Saved → {filename}")
    else:
        print(f"[✓] Found existing: {filename}")

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)

    # Style image — Van Gogh Starry Night (1280px is a valid Wikimedia size)
    download(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "images/style_starry_night.jpg"
    )

    # Content image — public domain sample
    download(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/640px-PNG_transparency_demonstration_1.png",
        "images/content.png"
    )

    print("\n[INFO] Running style transfer (may take a few minutes on CPU)...")
    content = load_image("images/content.png")
    style   = load_image("images/style_starry_night.jpg")

    output = run_style_transfer(content, style, num_steps=200)
    img = tensor_to_image(output)
    img.save("images/output.jpg")
    print("\n[✓] Done! Output saved to: images/output.jpg")