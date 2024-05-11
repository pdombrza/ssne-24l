import torch
import torch.utils
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from evaluator import train_evaluator, calculate_frechet_distance, get_distribution


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(3, 32, 5, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(3, 32, 5, stride=2, padding=1)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.LeakyReLU(self.fc_1(x))
        x = self.LeakyReLU(self.fc_2(x))
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc_1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.fc_1(x))
        h = self.LeakyReLU(self.fc_2(h))

        x_hat = torch.sigmoid(self.fc_3(h))
        x_hat = x_hat.view([-1, 3, 32, 32])
        return x_hat


class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)


    def reparameterization(self, mean, var):
        z = torch.randn_like(mean) * var + mean
        return z


    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var



def prepare_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def generate_reconstructions(model, input_images, device):
    model.eval()
    with torch.no_grad():
        reconst_imgs, means, log_var = model(input_images.to(device))
    reconst_imgs = reconst_imgs.cpu()
    return reconst_imgs


def generate_random_images(model, n_imgs, device):
    # Generate images
    model.eval()
    with torch.no_grad():
        generated_imgs = model.decoder(torch.randn([n_imgs, model.latent_dim]).to(device))
    generated_imgs = generated_imgs.cpu()
    return generated_imgs


def get_test_images(test_set, number):
    orig_data = [x[0] for x in list(test_set)[:number]]
    orig_data = torch.stack(orig_data)
    return orig_data


def visualize_images(input_images, generated_images, reconstruct=True):
    if reconstruct:
        imgs = torch.stack([input_images, generated_images], dim=1).flatten(0,1)
        title = "Reconstructions"
        grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=False, value_range=(-1,1))
    else:
        title = "Generations"
        grid = torchvision.utils.make_grid(generated_images, nrow=4, normalize=False, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    if len(input_images) == 4:
        plt.figure(figsize=(10,10))
    else:
        plt.figure(figsize=(15,10))
    plt.title(title)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()



def get_train_images(num, test_set):
    return torch.stack([test_set[i][0] for i in range(10,10+num)], dim=0)


def vae_loss(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD      = -0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def main():
    device = torch.device("cuda")
    prepare_cuda()
    data_path = "./trafic_32"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # load data
    batch_size = 256
    dataset = ImageFolder(data_path, transform=transform)
    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)
    class_amount = len(dataset.classes)

    # prep model
    vae =  VAE(latent_dim=128, hidden_dim=1024, x_dim=3072).to(device)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    train
    num_epochs = 3
    for n in range(num_epochs):
        losses_epoch = []
        for x, y in iter(trainloader):
            x = x.to(device)
            out, means, log_var = vae(x)
            loss = vae_loss(x, out, means, log_var)
            losses_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        L1_list = []
        for x, _ in iter(testloader):
            x  = x.to(device)
            out, _, _ = vae(x)
            L1_list.append(torch.mean(torch.abs(out-x)).item())
        print(f"Epoch {n} loss {np.mean(np.array(losses_epoch))}, test L1 = {np.mean(L1_list)}")
        scheduler.step()

    number = 1000
    test_images = get_test_images(test, number)

    # Reconstruct
    input_images = get_train_images(number, test_set=test)
    evaluator = train_evaluator(trainloader, 5, class_amount, device, 3*32*32, 256)
    reconst_images = generate_reconstructions(vae, input_images, device)
    # visualize_images(input_images, reconst_images)
    reconst_distance = calculate_frechet_distance(get_distribution(test_images, evaluator, device).numpy(), get_distribution(reconst_images, evaluator, device).numpy())
    print(f"Reconstruction fid: {reconst_distance}")

    # Generate
    gen_images = generate_random_images(vae, number, device)
    # visualize_images(input_images, gen_images, reconstruct=False)
    gen_distance = calculate_frechet_distance(get_distribution(test_images, evaluator, device).numpy(), get_distribution(gen_images, evaluator, device).numpy())
    print(f"Generation fid: {gen_distance}")





if __name__ == "__main__":
    main()