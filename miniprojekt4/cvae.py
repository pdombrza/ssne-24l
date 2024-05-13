import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(18)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(48)


        self.fc_1 = nn.Linear(12288, 6000)
        self.batch_norm5 = nn.BatchNorm1d(6000)
        self.fc_2 = nn.Linear(6000, latent_dim)
        self.batch_norm6 = nn.BatchNorm1d(latent_dim)
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        x = self.LeakyReLU(self.batch_norm1(self.conv1(x)))
        x = self.LeakyReLU(self.batch_norm2(self.conv2(x)))
        x = self.LeakyReLU(self.batch_norm3(self.conv3(x)))
        x = self.LeakyReLU(self.batch_norm4(self.conv4(x)))

        x = torch.flatten(x, 1)
        x = self.LeakyReLU(self.batch_norm5(self.fc_1(x)))
        x = self.LeakyReLU(self.batch_norm6(self.fc_2(x)))
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc_1 = nn.Linear(latent_dim, 6000)
        self.batch_norm1 = nn.BatchNorm1d(6000)
        self.fc_2 = nn.Linear(6000, 12288)
        self.batch_norm2 = nn.BatchNorm1d(12288)

        self.deconv1 = nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(18)
        self.deconv3 = nn.ConvTranspose2d(in_channels=18, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(9)
        self.deconv4 = nn.ConvTranspose2d(in_channels=9, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(3)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.batch_norm1(self.fc_1(x)))
        x = self.LeakyReLU(self.batch_norm2(self.fc_2(x)))

        x = x.view([-1, 48, 16, 16])
        x = self.LeakyReLU(self.batch_norm3(self.deconv1(x)))
        x = self.LeakyReLU(self.batch_norm4(self.deconv2(x)))
        x = self.LeakyReLU(self.batch_norm5(self.deconv3(x)))

        x_hat = torch.sigmoid(self.batch_norm6(self.deconv4(x)))
        return x_hat


class ConvVae(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)


    def reparameterization(self, mean, var):
        z = torch.randn_like(mean) * var + mean
        return z


    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
