import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers.models.unets.unet_2d import UNet2DModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

dataset = "cifar10"

####################################
# Set Random Seed for Reproducibility
####################################


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


####################################
# Define Model Classes
####################################


####################################
# Model Definitions (VAE, DDPM, DiffuseVAE)
####################################
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        channels = 3 if dataset == "cifar10" else 1
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        channels = 3 if dataset == "cifar10" else 1
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def create_unet(config):
    channels = 3 if dataset == "cifar10" else 1
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        dropout=0.3,
        attention_head_dim=8,
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


class DDPM:
    def __init__(self, model, timesteps=1000, device="cuda"):
        self.model = model
        self.timesteps = timesteps
        self.betas = self.linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]],
            dim=0,
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.device = device

    def linear_beta_schedule(self, timesteps, start=1e-4, end=0.02):
        return torch.linspace(start, end, timesteps)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[
            t
        ].view(-1, 1, 1, 1)
        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def predict_eps(self, x_t, t):
        return self.model(x_t, t)

    def p_sample(self, x_t, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[
            t
        ].view(-1, 1, 1, 1)
        pred_eps = self.predict_eps(x_t, t).sample
        one_div_sqrt_alpha_t = (1.0 / torch.sqrt(self.alphas[t])).view(
            -1, 1, 1, 1
        )
        mean = one_div_sqrt_alpha_t * (
            x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * pred_eps
        )
        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = 0.0
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alphas_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        var = betas_t * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.device)
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            x = self.p_sample(x, t)
        return x

    def training_loss(self, x_0):
        bsz = x_0.size(0)
        t = torch.randint(0, self.timesteps, (bsz,), device=self.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)
        pred = self.model(x_t, t).sample
        loss = nn.MSELoss()(pred, noise)
        return loss


class DiffuseVAE(nn.Module):
    def __init__(self, vae, ddpm, latent_dim=512):
        super().__init__()
        self.vae = vae
        self.ddpm = ddpm
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.vae(x)

    @torch.no_grad()
    def sample(self, num_samples=64):
        z = torch.randn(num_samples, self.latent_dim, device=self.ddpm.device)
        x_0 = self.vae.decoder(z)
        refined = self.ddpm.sample(x_0.shape)
        return refined

    def decode(self, z):
        return self.vae.decoder(z)

    def load_state_dict(self, state_dict):
        self.vae.load_state_dict(state_dict["vae"])
        self.ddpm.model.load_state_dict(state_dict["ddpm"])


####################################
# Load Saved Model
####################################
class ConfigDDPM:
    image_size = 32
    num_epochs = 500
    num_timesteps = 1000
    output_dir = "./diffuse_cifar10_stage2"


def load_model(config, device, real_loader=None, num_recon_samples=4):

    vae = VAE(latent_dim=512).to(device)
    unet = create_unet(config).to(device)
    ddpm = DDPM(unet, timesteps=config.num_timesteps, device=device)
    diffuse_vae = DiffuseVAE(vae, ddpm, latent_dim=512).to(device)

    # Load state dict
    model_path = os.path.join(
        config.output_dir,
        "models",
        f"diffuse_vae_epoch_{config.num_epochs}.pth",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    diffuse_vae.load_state_dict(torch.load(model_path, map_location=device))
    diffuse_vae.eval()
    print(f"Loaded model from {model_path}")

    if real_loader is not None:
        try:
            real_batch, _ = next(iter(real_loader))
        except StopIteration:
            raise ValueError("The provided real_loader is empty.")

        real_batch = real_batch.to(device)

        with torch.no_grad():
            recon_batch, _, _ = diffuse_vae.vae(real_batch)

        real_samples = real_batch[:num_recon_samples]
        recon_samples = recon_batch[:num_recon_samples]

        comparison = torch.cat([real_samples, recon_samples], dim=0)

        comparison_path = os.path.join(
            config.output_dir, "models", "reconstructions.png"
        )

        vutils.save_image(
            comparison,
            comparison_path,
            nrow=num_recon_samples,
            normalize=True,
            padding=2,
            pad_value=1,
        )
        print(f"Reconstructed images saved to {comparison_path}")

    return diffuse_vae


####################################
# Prepare Data for Metrics
####################################


def get_real_images(dataloader, num_images, device):
    real_images = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            real_images.append(imgs)
            if len(real_images) * imgs.size(0) >= num_images:
                break
    real_images = torch.cat(real_images, dim=0)[:num_images]
    return real_images


def preprocess_images(images):
    # Denormalize from [-1,1] to [0,1]
    images = (images * 0.5) + 0.5
    # Resize to 299x299
    images = nn.functional.interpolate(
        images, size=(299, 299), mode="bilinear", align_corners=False
    )
    if dataset != "cifar10":
        # Convert to 3 channels by repeating the grayscale channel
        images = images.repeat(1, 3, 1, 1)
    # Scale to [0,255] and convert to uint8
    images = (images * 255).clamp(0, 255).to(torch.uint8)
    return images


####################################
# Compute Metrics
####################################


def compute_metrics(
    diffuse_vae,
    real_loader,
    device,
    num_samples=1000,
    batch_size=50,
    comparison_save_path="visual_metrics_comparison.png",
):

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception_score = InceptionScore().to(device)

    real_images = []
    for imgs, _ in real_loader:
        imgs = imgs.to(device)
        real_images.append(imgs)
        if len(real_images) * imgs.size(0) >= num_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    real_images = preprocess_images(real_images)
    fid.update(real_images, real=True)

    comparison_real = []
    comparison_fake = []

    num_batches = num_samples // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Generating for Metrics"):
        with torch.no_grad():
            samples = diffuse_vae.sample(num_samples=batch_size)
            samples = samples.clamp(-1, 1)
            samples = preprocess_images(samples)
            fid.update(samples, real=False)
            inception_score.update(samples)

    # Compute FID
    fid_score = fid.compute().item()
    print(f"Frechet Inception Distance (FID): {fid_score:.4f}")

    # Compute IS
    is_score, is_std = inception_score.compute()
    print(f"Inception Score (IS): {is_score:.4f} ± {is_std:.4f}")

    return fid_score, (is_score, is_std)


####################################
# Latent Space Visualization
####################################


def visualize_latent_space_pca(
    diffuse_vae,
    dataset,
    device,
    sample_fraction=0.1,
    save_path="latent_space_pca.png",
):

    num_samples = int(len(dataset) * sample_fraction)
    subset_indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    mu_list = []
    labels_list = []
    diffuse_vae.vae.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            mu, _ = diffuse_vae.vae.encoder(imgs)
            mu_list.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
    mu = np.concatenate(mu_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Dimensionality reduction (PCA)
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(mu)

    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        mu_2d[:, 0], mu_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10
    )
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title("Latent Space Visualization (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"PCA latent space visualization saved as '{save_path}'")


def visualize_latent_space_tsne(
    diffuse_vae,
    dataset,
    device,
    sample_fraction=0.1,
    save_path="latent_space_tsne.png",
):

    num_samples = int(len(dataset) * sample_fraction)
    subset_indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    # Encode images to latent space
    mu_list = []
    labels_list = []
    diffuse_vae.vae.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            mu, _ = diffuse_vae.vae.encoder(imgs)
            mu_list.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
    mu = np.concatenate(mu_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Dimensionality reduction (t-SNE)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    mu_2d = tsne.fit_transform(mu)

    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        mu_2d[:, 0], mu_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10
    )
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"t-SNE latent space visualization saved as '{save_path}'")


####################################
# Main Execution
####################################


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ConfigDDPM()

    os.makedirs(os.path.join(config.output_dir, "models"), exist_ok=True)
    os.makedirs(
        os.path.join(config.output_dir, "visualizations"), exist_ok=True
    )

    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    if dataset == "cifar10":
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True
        )
    else:
        test_dataset = datasets.MNIST(
            root="./data", train=False, transform=transform, download=True
        )
    test_loader = DataLoader(
        test_dataset, batch_size=4, shuffle=True, num_workers=4
    )

    # Load model and perform reconstruction
    diffuse_vae = load_model(
        config, device, real_loader=test_loader, num_recon_samples=4
    )

    # Compute Metrics
    fid_score, is_score = compute_metrics(
        diffuse_vae, test_loader, device, num_samples=1000, batch_size=50
    )

    # Visualize Latent Space using PCA
    visualize_latent_space_pca(
        diffuse_vae,
        test_dataset,
        device,
        sample_fraction=1,
        save_path=os.path.join(
            config.output_dir, "visualizations", "latent_space_pca.png"
        ),
    )

    # Visualize Latent Space using t-SNE
    visualize_latent_space_tsne(
        diffuse_vae,
        test_dataset,
        device,
        sample_fraction=1,
        save_path=os.path.join(
            config.output_dir, "visualizations", "latent_space_tsne.png"
        ),
    )


if __name__ == "__main__":
    main()
