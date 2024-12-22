import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from tqdm import tqdm


####################################
# VAE Model Components
####################################
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # CIFAR-10: image size 32x32, 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
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
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
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


# Configuration Class
class TrainingConfigDDPM:
    image_size = (
        32  # 생성되는 이미지 해상도 (CIFAR-10 = 32*32 , CelebA-64 = 64*64)
    )
    train_batch_size = 128
    eval_batch_size = 128
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"
    output_dir = "./diffuse_cifar10"
    seed = 0
    num_timesteps = 1000
    checkpoint_dir = "./diffuse_checkpoints"


# Model Definition
def create_unet(config):
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        dropout=0.3,
        attention_head_dim=8,
        down_block_types=(
            "DownBlock2D",  # 32 -> 16
            "AttnDownBlock2D",  # 16 -> 8
            "DownBlock2D",  # 8 -> 4
            "DownBlock2D",  # 4 -> 2
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",  # 대응하는 UpBlock2D 대신 여기서도 attention 쌍을 맞추려면 "AttnUpBlock2D"를 16x16 해당 스케일에서 넣어줄 수 있음
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


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
        """q(x_t | x_0)"""
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
        # model predicts epsilon given x_t and t
        return self.model(x_t, t)

    def p_sample(self, x_t, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[
            t
        ].view(-1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alphas_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)

        pred_eps = self.predict_eps(x_t, t).sample
        # p(x_{t-1}|x_t)
        one_div_sqrt_alpha_t = (1.0 / torch.sqrt(self.alphas[t])).view(
            -1, 1, 1, 1
        )
        mean = one_div_sqrt_alpha_t * (
            x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * pred_eps
        )
        # mean = 1./torch.sqrt(self.alphas[t])*(x_t - betas_t/sqrt_one_minus_alphas_cumprod_t * pred_eps)
        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = 0.0
        var = betas_t * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            # t_tensor = torch.tensor([t]*shape[0], device=self.device)
            # x = self.p_sample(x, t_tensor)
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


####################################
# DiffuseVAE: VAE + DDPM 결합 예시
####################################
class DiffuseVAE(nn.Module):
    def __init__(self, vae, ddpm, latent_dim=512):
        super().__init__()
        self.vae = vae
        self.ddpm = ddpm
        self.latent_dim = latent_dim

    def forward(self, x):
        # VAE로부터 z 샘플 후 reconstruction
        recon, mu, logvar = self.vae(x)
        return recon, mu, logvar

    def sample(self, num_samples=64):
        # 랜덤한 latent vector로부터 샘플링
        z = torch.randn(num_samples, self.latent_dim, device=self.ddpm.device)
        x_0 = self.vae.decoder(z)
        # DDPM을 통해 refinement (denoising) 수행
        # 여기선 x_0를 초기값으로 하여, 다시 noise를 추가한 뒤 샘플링
        # 실제로는 별도 조건이나 forward-backward 과정을 조정 가능
        with torch.no_grad():
            # 간단하게 DDPM forward를 적용한 후 reverse sampling
            t = torch.randint(
                0, self.ddpm.timesteps, (num_samples,), device=self.ddpm.device
            ).long()
            noisy = self.ddpm.q_sample(x_0, t)
            # noisy로부터 DDPM reverse sampling
            # 여기서는 t=0부터 진행은 어렵고, 재시작 개념 단순 예시
            # 실제론 DDPM의 샘플링 로직을 z에서부터 진행하는 것이 일반적
            # 여기서는 VAE 결과물을 약간 노이즈주고 DDPM으로 denoise하는 개념적 예시
            refined = self.ddpm.sample(x_0.shape)
            return refined

    def state_dict(self):
        return {
            "vae": self.vae.state_dict(),
            "ddpm": self.ddpm.model.state_dict(),
        }


def vae_loss_function(recon, x, mu, logvar, kl_weight=1.0):
    recon_loss = nn.MSELoss()(recon, x)
    # KLD
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kld, recon_loss, kld


def load_dataset(config):
    ####################################
    # CIFAR-10 Dataset
    ####################################
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
    )
    return train_loader


# 1. VAE 사전학습
def train_vae(device, config):
    train_loader = load_dataset(config)

    latent_dim = 512
    vae_pretrain = VAE(latent_dim=latent_dim).to(device)
    optimizer_vae_pretrain = optim.Adam(vae_pretrain.parameters(), lr=1e-4)

    print("VAE pretrain started")
    pretrain_epochs = 500
    for epoch in tqdm(range(pretrain_epochs), desc="Epoch", position=1):
        vae_pretrain.train()
        total_vae_loss = 0.0
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer_vae_pretrain.zero_grad()
            recon, mu, logvar = vae_pretrain(imgs)
            vae_loss, _, _ = vae_loss_function(
                recon, imgs, mu, logvar, kl_weight=1.0
            )
            vae_loss.backward()
            optimizer_vae_pretrain.step()
            total_vae_loss += vae_loss.item()
        print(
            f"VAE Pretrain Epoch [{epoch}/{pretrain_epochs}] Loss: {total_vae_loss/len(train_loader):.4f}"
        )

    # VAE 파라미터 저장
    torch.save(
        vae_pretrain.state_dict(),
        f"vae_mnist_pretrained_epoch{pretrain_epochs}.pth",
    )


def save_model(model, epoch, config):
    os.makedirs(os.path.join(config.output_dir, "models"), exist_ok=True)
    checkpoint_path = os.path.join(
        config.output_dir, f"models/diffuse_vae_epoch_{epoch}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved: diffuse_vae_epoch_{epoch}")


def load_model(unet, config, device, latent_dim=512, model_epoch=500):
    # 동일한 모델 구조를 다시 생성
    loaded_diffuse_vae = DiffuseVAE(
        vae=VAE(latent_dim=latent_dim),
        ddpm=DDPM(model=unet, timesteps=config.num_timesteps, device=device),
        latent_dim=latent_dim,
    ).to(device)
    # 파라미터 로드
    epoch = model_epoch
    checkpoint_path = os.path.join(
        config.output_dir, f"models/diffuse_vae_epoch_{epoch}.pth"
    )
    loaded_diffuse_vae.load_state_dict(torch.load(checkpoint_path))
    loaded_diffuse_vae.eval()
    return loaded_diffuse_vae


def save_image(epoch, diffuse_vae, config):
    # 샘플링 테스트
    with torch.no_grad():
        samples = diffuse_vae.sample(num_samples=16)
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
        image_path = os.path.join(
            config.output_dir, f"samples/mnist_samples_epoch_{epoch}.png"
        )
        vutils.save_image((samples * 0.5 + 0.5), image_path, nrow=4)


def train_diffuse_vae(device, ddpm_config):
    train_loader = load_dataset(ddpm_config)

    latent_dim = 512

    # 2. 사전학습된 VAE를 로드하고 DiffuseVAE 구성
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("./vae_mnist_pretrained_epoch500.pth"))
    vae.eval()

    unet = create_unet(ddpm_config).to(device)
    ddpm = DDPM(unet, timesteps=ddpm_config.num_timesteps, device=device)
    diffuse_vae = DiffuseVAE(vae, ddpm, latent_dim).to(device)
    # checkpoint = torch.load("./models/diffuse_vae_epoch_50.pth")
    # vae.load_state_dict(checkpoint["vae"])
    # vae.eval()
    # ddpm.model.load_state_dict(checkpoint["ddpm"])

    # 이제 VAE는 이미 학습된 상태이므로 VAE 파라미터 업데이트 비율을 낮추거나 freeze할 수 있음
    # 예: VAE 파라미터 freeze
    for param in diffuse_vae.vae.parameters():
        param.requires_grad = False

    optimizer_ddpm = optim.Adam(diffuse_vae.ddpm.model.parameters(), lr=2e-4)

    ####################################
    # Training Loop
    ####################################
    start_epoch = 0
    epochs = 500

    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch", position=1):
        diffuse_vae.train()
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer_ddpm.zero_grad()
            recon, mu, logvar = diffuse_vae(imgs)
            ddpm_loss = diffuse_vae.ddpm.training_loss(recon)
            ddpm_loss.backward()
            optimizer_ddpm.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{i}/{len(train_loader)}]: DDPM_loss:{ddpm_loss.item():.4f}"
                )

        if (epoch < 100) or ((epoch + 1) % ddpm_config.save_image_epochs == 0):
            save_image(epoch + 1, diffuse_vae, ddpm_config)
        if (epoch + 1) % ddpm_config.save_model_epochs == 0:
            save_model(diffuse_vae, epoch + 1, ddpm_config)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm_config = TrainingConfigDDPM()
    train_vae(device, ddpm_config)
    train_diffuse_vae(device, ddpm_config)
