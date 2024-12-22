import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers.models.unets.unet_2d import UNet2DModel
from torch.utils.data import DataLoader
from tqdm import tqdm

####################################
# Dataset (MNIST or CIFAR10)
####################################
dataset = "mnist"
transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
if dataset == "cifar10":
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
else:
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4
)


####################################
# Flow 기반 Prior 구현 (예: RealNVP)
####################################
class RealNVP(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=256, num_flows=2):
        super().__init__()
        self.flows = nn.ModuleList(
            [RealNVPBlock(latent_dim, hidden_dim) for _ in range(num_flows)]
        )

    def forward(self, z, reverse=False):
        if not reverse:
            # forward flow: z0 ~ N(0,I) -> zK
            for flow in self.flows:
                z = flow(z)
        else:
            # inverse flow: zK -> z0
            for flow in reversed(self.flows):
                z = flow(z, reverse=True)
        return z


class RealNVPBlock(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.t = nn.Sequential(
            nn.Linear(latent_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim // 2),
        )
        self.s = nn.Sequential(
            nn.Linear(latent_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim // 2),
        )

    def forward(self, z, reverse=False):
        z1, z2 = z.chunk(2, dim=1)
        if not reverse:
            # forward
            shift = self.t(z1)
            scale = self.s(z1)
            z2 = z2 * torch.exp(scale) + shift
            return torch.cat([z1, z2], dim=1)
        else:
            # inverse
            z1, z2 = z.chunk(2, dim=1)
            shift = self.t(z1)
            scale = self.s(z1)
            z2 = (z2 - shift) * torch.exp(-scale)
            return torch.cat([z1, z2], dim=1)


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
        self.flow = RealNVP(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = self.flow(z)
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class TrainingConfigDDPM:
    image_size = 32
    num_epochs = 500
    num_timesteps = 1000
    output_dir = "./diffuse_mnist_stage3"
    save_image_epochs = 10
    save_model_epochs = 100


def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


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
        self.betas = linear_beta_schedule(timesteps).to(device)
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
        for t in reversed(range(self.timesteps)):
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

    def state_dict(self):
        return {
            "vae": self.vae.state_dict(),
            "ddpm": self.ddpm.model.state_dict(),
        }


def hybrid_vae_loss(recon, x, mu, logvar, kl_weight=1.0):
    mse_loss = nn.MSELoss()(recon, x)
    l1_loss = nn.L1Loss()(recon, x)
    recon_loss = 0.5 * (mse_loss + l1_loss)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kld, recon_loss, kld


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainingConfigDDPM()

    vae = VAE(latent_dim=512).to(device)
    unet = create_unet(config).to(device)  # type: ignore
    ddpm = DDPM(unet, timesteps=config.num_timesteps, device=device)
    diffuse_vae = DiffuseVAE(vae, ddpm, latent_dim=512).to(device)

    # Joint training with weighting
    lambda_vae = 1.0
    lambda_ddpm_base = 0.5

    optimizer_vae = optim.Adam(diffuse_vae.vae.parameters(), lr=1e-4)
    optimizer_ddpm = optim.Adam(diffuse_vae.ddpm.model.parameters(), lr=2e-4)

    os.makedirs(os.path.join(config.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)

    for epoch in range(config.num_epochs):
        diffuse_vae.train()
        total_loss = 0
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)

            # Forward
            recon, mu, logvar = diffuse_vae(imgs)
            vae_loss, recon_loss, kld_loss = hybrid_vae_loss(
                recon, imgs, mu, logvar, kl_weight=1.0
            )

            variance = logvar.exp().mean()
            lambda_ddpm = lambda_ddpm_base * (1.0 + variance.item())

            ddpm_loss = diffuse_vae.ddpm.training_loss(recon)
            total = lambda_vae * vae_loss + lambda_ddpm * ddpm_loss

            optimizer_vae.zero_grad()
            optimizer_ddpm.zero_grad()
            total.backward()
            optimizer_vae.step()
            optimizer_ddpm.step()

            total_loss += total.item()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{config.num_epochs}] Step[{i}/{len(train_loader)}]: "
                    f"VAE_loss: {vae_loss.item():.4f}, DDPM_loss: {ddpm_loss.item():.4f}, total: {total.item():.4f}, variance: {variance.item():.4f}"
                )

        if (epoch < 100) or ((epoch + 1) % config.save_image_epochs == 0):
            diffuse_vae.eval()
            with torch.no_grad():
                samples = diffuse_vae.sample(num_samples=16)
                image_path = os.path.join(
                    config.output_dir,
                    f"samples/mnist_samples_epoch_{epoch+1}.png",
                )
                vutils.save_image((samples * 0.5 + 0.5), image_path, nrow=4)

        # Save model
        if (epoch + 1) % config.save_model_epochs == 0:
            model_path = os.path.join(
                config.output_dir, f"models/diffuse_vae_epoch_{epoch+1}.pth"
            )
            torch.save(diffuse_vae.state_dict(), model_path)
            print(f"Model saved at epoch {epoch+1}: {model_path}")


if __name__ == "__main__":
    main()
