import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 2512)
        self.batch_norm1 = nn.BatchNorm1d(2512)
        self.fc_2 = nn.Linear(2512, 2048)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.fc_3 = nn.Linear(2048, 1612)
        self.batch_norm3 = nn.BatchNorm1d(1612)
        self.fc_4 = nn.Linear(1612, 1024)
        self.batch_norm4 = nn.BatchNorm1d(1024)
        self.fc_5 = nn.Linear(1024, 848)
        self.batch_norm5 = nn.BatchNorm1d(848)
        self.fc_6 = nn.Linear(848, 612)
        self.batch_norm6 = nn.BatchNorm1d(612)
        self.fc_7 = nn.Linear(612, 428)
        self.batch_norm7 = nn.BatchNorm1d(428)
        self.fc_8 = nn.Linear(428, 256)
        self.batch_norm8 = nn.BatchNorm1d(256)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.batch_norm1(self.LeakyReLU(self.fc_1(x)))
        x = self.batch_norm2(self.LeakyReLU(self.fc_2(x)))
        x = self.batch_norm3(self.LeakyReLU(self.fc_3(x)))
        x = self.batch_norm4(self.LeakyReLU(self.fc_4(x)))
        x = self.batch_norm5(self.LeakyReLU(self.fc_5(x)))
        x = self.batch_norm6(self.LeakyReLU(self.fc_6(x)))
        x = self.batch_norm7(self.LeakyReLU(self.fc_7(x)))
        x = self.batch_norm8(self.LeakyReLU(self.fc_8(x)))
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc_1 = nn.Linear(latent_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc_2 = nn.Linear(256, 428)
        self.batch_norm2 = nn.BatchNorm1d(428)
        self.fc_3 = nn.Linear(428, 612)
        self.batch_norm3 = nn.BatchNorm1d(612)
        self.fc_4 = nn.Linear(612, 848)
        self.batch_norm4 = nn.BatchNorm1d(848)
        self.fc_5 = nn.Linear(848, 1024)
        self.batch_norm5 = nn.BatchNorm1d(1024)
        self.fc_6 = nn.Linear(1024, 1612)
        self.batch_norm6 = nn.BatchNorm1d(1612)
        self.fc_7 = nn.Linear(1612, 2048)
        self.batch_norm7 = nn.BatchNorm1d(2048)
        self.fc_8 = nn.Linear(2048, 2512)
        self.batch_norm8 = nn.BatchNorm1d(2512)
        self.fc_9 = nn.Linear(2512, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.batch_norm1(self.LeakyReLU(self.fc_1(x)))
        x = self.batch_norm2(self.LeakyReLU(self.fc_2(x)))
        x = self.batch_norm3(self.LeakyReLU(self.fc_3(x)))
        x = self.batch_norm4(self.LeakyReLU(self.fc_4(x)))
        x = self.batch_norm5(self.LeakyReLU(self.fc_5(x)))
        x = self.batch_norm6(self.LeakyReLU(self.fc_6(x)))
        x = self.batch_norm7(self.LeakyReLU(self.fc_7(x)))
        x = self.batch_norm8(self.LeakyReLU(self.fc_8(x)))

        # x = self.LeakyReLU(self.fc_1(x))
        # x = self.LeakyReLU(self.fc_2(x))
        # x = self.LeakyReLU(self.fc_3(x))
        # x = self.LeakyReLU(self.fc_4(x))
        # x = self.LeakyReLU(self.fc_5(x))
        # x = self.LeakyReLU(self.fc_6(x))
        # x = self.LeakyReLU(self.fc_7(x))
        # x = self.LeakyReLU(self.fc_8(x))
        x_hat = torch.sigmoid(self.fc_9(x))
        x_hat = x_hat.view([-1, 3, 32, 32])
        return x_hat


class ExtendedVAE(nn.Module):
    def __init__(self, x_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim=x_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=x_dim)


    def reparameterization(self, mean, var):
        z = torch.randn_like(mean) * var + mean
        return z


    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
