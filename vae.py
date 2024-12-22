import platform

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers.models.autoencoders.vae import Decoder, Encoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# CIFAR에 맞춘 VAE 모델 정의
class VAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        block_out_channels: tuple[int, ...] | None = None,
    ):
        super(VAE, self).__init__()

        if block_out_channels is None:
            block_out_channels = (32, 64, 128, 128)

        # 인코더
        self.encoder = nn.Sequential(
            Encoder(
                in_channels=input_channels,
                out_channels=latent_dim,
                down_block_types=tuple(
                    "DownEncoderBlock2D" for _ in range(len(block_out_channels))
                ),
                block_out_channels=block_out_channels,
            ),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(latent_dim * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 4 * 4, latent_dim)

        # 디코더
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            Decoder(
                in_channels=latent_dim,
                out_channels=input_channels,
                up_block_types=tuple(
                    "UpDecoderBlock2D" for _ in range(len(block_out_channels))
                ),
                block_out_channels=block_out_channels[::-1],
            ),
            nn.Sigmoid(),
        )

        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h[:, : self.latent_dim * 4 * 4])
        logvar = self.fc_logvar(h[:, self.latent_dim * 4 * 4 :])
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = self.fc_decode(z).view(-1, 128, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# VAE 손실 함수 정의
def vae_loss(recon_x, x, mu, logvar):
    # 재구성 손실 (Binary Cross-Entropy)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    # KL 발산 손실
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


# 데이터 준비 (CIFAR-10)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 모델, 옵티마이저 및 학습 설정
if platform.system() == "Darwin":
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_channels=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습 루프
epochs = 100
for epoch in tqdm(range(epochs), desc="Epoch", position=1):
    model.train()
    train_loss = 0
    for x, _ in tqdm(train_loader, desc="Steps", position=0, leave=False):
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(
        f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}"
    )

# 학습 완료 후 샘플링
model.eval()
with torch.no_grad():
    z = torch.randn(16, 128).to(device)  # 잠재 공간에서 샘플 생성
    samples = (
        model.decode(z).cpu().permute(0, 2, 3, 1)
    )  # [N, C, H, W] -> [N, H, W, C]

# 시각화 (샘플 출력)
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axs.flat):
    ax.imshow(samples[i].numpy())
    ax.axis("off")
plt.show()

# 모델 저장
torch.save(model.state_dict(), "vae_cifar10.pth")

