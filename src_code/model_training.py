import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pytorch_msssim import ms_ssim, MS_SSIM
from torchvision.utils import save_image
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F

from xception_dcnn import BottleneckAutoencoder 


# Loss weighting factors
lambda_l1 = 0.4  # Weight for the L1 (pixel-wise) loss
lambda_ssim = 0.6 # Weight for the 1-MS_SSIM (structural) loss

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")

data_dir = "/home/chinasa/python_projects/denoising/images/idiap/spoofed"

# Define the safe, isolated output root first
output_root = "/home/chinasa/python_projects/auto_encoder/outputs"
os.makedirs(output_root, exist_ok=True)

# Define the model save path inisde the isolated output root
save_path = os.path.join(output_root, "ssmbottleneck_autoencoder2.pth")

# Define the sample directory INSIDE the isolated output root
#Use output_root directly for consistency
sample_dir = os.path.join(output_root, 'reconstruction_samples2')
os.makedirs(sample_dir, exist_ok=True)

sample_interval = 5

batch_size = 2
epochs = 100
lr = 1e-5
weight_decay = 1e-5
patience = 5


# DATASET
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)


bottleneck_size = 3000

model = BottleneckAutoencoder(bottleneck_features=bottleneck_size, out_ch=1).to(device)
l1_criterion = nn.L1Loss() # reconstruction loss
ssim_criterion = MS_SSIM(data_range=2.0, size_average=True, channel=1)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

#training
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = None

# Get a fixed batch of images for visualization
fixed_batch = next(iter(val_loader))
fixed_imgs, _ = fixed_batch
fixed_imgs = fixed_imgs.to(device)


for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs) # outputs is the reconstructed image X'
        l1_loss = l1_criterion(outputs, imgs) # Loss = ||X - X'||^2
        ssim_value = ssim_criterion(outputs, imgs)
        ssim_loss = 1.0 - ssim_value
        loss = (lambda_l1 * l1_loss) + (lambda_ssim * ssim_loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    #validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            l1_loss = l1_criterion(outputs, imgs)
            ssim_value = ssim_criterion(outputs, imgs)
            ssim_loss = 1.0 - ssim_value
            loss = (lambda_l1 * l1_loss) + (lambda_ssim * ssim_loss)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)

    if (epoch + 1) % sample_interval == 0:
        model.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            # Get the model's output for the fixed set of images
            reconstructed_imgs = model(fixed_imgs)

            # Combine original and reconstructed images side-by-side
            # The fixed_imgs and reconstructed_imgs are normalized [-1, 1], 
            # so we un-normalize them before saving.
            # (X * std) + mean
            
            # Since your normalization is [0.5], [0.5], the reverse is:
            # (X * 0.5) + 0.5 = (X + 1) / 2
            
            comparison = torch.cat((fixed_imgs, reconstructed_imgs), dim=0)
            
            # Save the grid of images (originals on top, reconstructions on bottom)
            save_image(
                comparison * 0.5 + 0.5, # Un-normalize from [-1, 1] to [0, 1]
                os.path.join(sample_dir, f'epoch_{epoch+1:03d}.png'),
                nrow=batch_size # Set number of images per row
            )
            print(f"🖼️ Saved reconstruction sample for epoch {epoch+1}.")

    # --- Scheduler & Early Stopping ---
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_wts = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Load best weights
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)

# --------------------- SAVE --------------------- #

torch.save(model.state_dict(), save_path) 
print(f"✅ Model saved successfully to {save_path}.")
