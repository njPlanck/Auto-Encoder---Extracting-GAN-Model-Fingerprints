import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from xception_dcnn import BottleneckAutoencoder  # same class as in training

# ---------------- CONFIG ---------------- #
model_path = "/home/chinasa/python_projects/auto_encoder/outputs/ssmbottleneck_autoencoder2.pth"
input_root = "/home/chinasa/python_projects/denoising/images/idiap/synthetic/"
output_root = "/home/chinasa/python_projects/auto_encoder/outputs/images/idiap/"
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")

# ---------------- TRANSFORMS ---------------- #
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def denormalize(tensor):
    """Reverse normalization from [-1, 1] → [0, 1]"""
    return tensor * 0.5 + 0.5

# ---------------- AUTO-DETECT BOTTLENECK SIZE ---------------- #
checkpoint = torch.load(model_path, map_location=device)
if "bottleneck.2.weight" in checkpoint:
    bottleneck_size = checkpoint["bottleneck.2.weight"].shape[0]
else:
    bottleneck_size = list(checkpoint.values())[0].shape[0]
print(f"🔍 Detected bottleneck size: {bottleneck_size}")

# ---------------- LOAD MODEL ---------------- #
model = BottleneckAutoencoder(bottleneck_features=bottleneck_size, out_ch=1).to(device)
model.load_state_dict(checkpoint)
model.eval()
print("✅ Model loaded successfully.")

# ---------------- PROCESS SUBFOLDERS ---------------- #
valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

with torch.no_grad():
    # Loop through GAN subfolders (cycleGAN, distanceGAN, etc.)
    for subfolder in sorted(os.listdir(input_root)):
        input_dir = os.path.join(input_root, subfolder)
        if not os.path.isdir(input_dir):
            continue  # skip non-folder files

        output_dir = os.path.join(output_root, subfolder)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🚀 Processing folder: {subfolder}")

        # Process all images in this subfolder
        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(valid_exts):
                continue

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load and preprocess image
            img = Image.open(input_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Forward pass
            output_tensor = model(img_tensor)

            # Denormalize and save
            output_img = denormalize(output_tensor.squeeze(0))
            save_image(output_img, output_path)

            print(f"  ✅ Denoised: {subfolder}/{filename}")

print("\n🎉 All images denoised and saved successfully, preserving folder structure!")
