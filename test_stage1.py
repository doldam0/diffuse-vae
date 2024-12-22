import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
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
    save_model_epochs = 100
    mixed_precision = "fp16"
    output_dir = "./diffuse_cifar10"
    seed = 0
    num_timesteps = 1000
    checkpoint_dir = "./diffuse_checkpoints"


ddpm_config = TrainingConfigDDPM()


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

    def load_state_dict(self, state_dict, strict=True):
        self.vae.load_state_dict(state_dict["vae"], strict=strict)
        self.ddpm.load_state_dict(state_dict["ddpm"], strict=strict)


def preprocess_images(images):
    """
    이미지 전처리: [-1, 1] -> [0, 255], 299x299 리사이즈, 3채널 변환, uint8 타입
    """
    # [-1, 1] 범위를 [0, 1]로 변환
    images = images * 0.5 + 0.5  # [0, 1] 범위로 변환
    images = images.to(torch.float32)  # float32로 변환
    images = F.interpolate(
        images, size=(299, 299), mode="bilinear", align_corners=False
    )  # 리사이즈
    images = (
        (images * 255).clamp(0, 255).to(torch.uint8)
    )  # [0, 255]로 스케일링 후 uint8로 변환
    images = images.repeat(1, 3, 1, 1)  # 1채널 -> 3채널 복제
    return images


######################################
# Evaluation metrics
######################################


def evaluate_model(model, real_loader, device, num_samples=1000, batch_size=50):
    model.eval()
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception_score = InceptionScore(normalize=True).to(device)

    # Real images 준비
    real_images = []
    for imgs, _ in real_loader:
        imgs = imgs.to(device)
        real_images.append(imgs)
        if len(real_images) * imgs.size(0) >= num_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    real_images = preprocess_images(real_images)
    fid.update(real_images, real=True)

    # Fake images 생성 및 평가
    for _ in tqdm(range(num_samples // batch_size), desc="Generating Samples"):
        with torch.no_grad():
            samples = model.sample(num_samples=batch_size).clamp(-1, 1)
            samples = preprocess_images(samples)
            fid.update(samples, real=False)
            inception_score.update(samples)

    # FID 및 IS 계산
    fid_value = fid.compute().item()
    is_mean, is_std = inception_score.compute()
    print(f"FID: {fid_value:.4f}")
    print(f"IS: {is_mean:.4f} ± {is_std:.4f}")

    return fid_value, is_mean, is_std


######################################
# Model Load & Eval
######################################


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 512

    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )

    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("./vae_mnist_pretrained_epoch500.pth"))
    vae.eval()

    unet = UNet2DModel(
        sample_size=32,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        dropout=0.3,
    ).to(device)
    ddpm = DDPM(unet, timesteps=1000, device=device)
    diffuse_vae = DiffuseVAE(vae, ddpm, latent_dim).to(device)

    model_path = "./diffuse_cifar10/models/diffuse_vae_epoch_500.pth"
    diffuse_vae.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")

    fid, is_mean, is_std = evaluate_model(
        diffuse_vae, test_loader, device, num_samples=1000, batch_size=50
    )
    print(
        f"Final Evaluation -> FID: {fid:.4f}, IS: {is_mean:.4f} ± {is_std:.4f}"
    )


if __name__ == "__main__":
    main()
