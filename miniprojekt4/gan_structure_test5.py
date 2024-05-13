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
from evaluator import train_evaluator, calculate_frechet_distance, get_distribution, eval_evaluator

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_4 = nn.Linear(128, 128)
        self.fc_5 = nn.Linear(128, 128)
        self.fc_6 = nn.Linear(128, 128)
        self.fc_7 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.ReLU(self.fc_1(x))
        x = self.ReLU(self.fc_2(x))
        x = self.ReLU(self.fc_3(x))
        x = self.ReLU(self.fc_4(x))
        x = self.ReLU(self.fc_5(x))
        x = self.ReLU(self.fc_6(x))
        x = self.ReLU(self.fc_7(x))
        x = self.fc_out(x)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.fc_1 = nn.Linear(latent_dim, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 256)
        self.fc_4 = nn.Linear(256, 512)
        self.fc_5 = nn.Linear(512, 256)
        self.fc_6 = nn.Linear(256, 128)
        self.fc_7 = nn.Linear(128, 128)
        self.fc_8 = nn.Linear(128, 128)
        self.fc_9 = nn.Linear(128, 128)
        self.fc_10 = nn.Linear(128, output_dim)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        h = self.ReLU(self.fc_1(x))
        h = self.ReLU(self.fc_2(h))
        h = self.ReLU(self.fc_3(h))
        h = self.ReLU(self.fc_4(h))
        h = self.ReLU(self.fc_5(h))
        h = self.ReLU(self.fc_6(h))
        h = self.ReLU(self.fc_7(h))
        h = self.ReLU(self.fc_8(h))
        h = self.ReLU(self.fc_9(h))
        x_hat = torch.sigmoid(self.fc_10(h))
        x_hat = x_hat.view([-1, 3, 32, 32])
        return x_hat

def prepare_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

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
        grid = torchvision.utils.make_grid(generated_images, nrow=min(len(input_images), 100), normalize=False, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    if len(input_images) == 4:
        plt.figure(figsize=(10,10))
    else:
        plt.figure(figsize=(100,100))
    plt.title(title)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def get_train_images(num, test_set):
    return torch.stack([test_set[i][0] for i in range(10,10+num)], dim=0)

def generate_random_images(model, latent_dim,  n_imgs, device):
    model.eval()
    with torch.no_grad():
        generated_imgs = model(torch.randn(n_imgs, latent_dim, device=device))
    generated_imgs = generated_imgs.cpu()
    return generated_imgs

def main():
    device = torch.device("cuda")
    prepare_cuda()
    data_path = "./trafic_32"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    batch_size = 256
    dataset = ImageFolder(data_path, transform=transform)
    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=8)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)
    class_amount = len(dataset.classes)

    latent_dim = 32
    generator = Generator(latent_dim=latent_dim, output_dim=3072).to(device)
    discriminator = Discriminator(input_dim=3072).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    generator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=generator_optimizer, gamma=0.99)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=discriminator_optimizer, gamma=0.99)

    criterion = nn.MSELoss()

    G_losses = []
    D_losses = []
    num_epochs = 100
    for epoch in range(num_epochs):
        discriminator_fake_acc = []
        discriminator_real_acc = []
        for i, data in enumerate(trainloader, 0):

            discriminator_optimizer.zero_grad()
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            label = torch.ones((b_size,), dtype=torch.float, device=device)
            output = discriminator(real_images).view(-1)
            error_discriminator_real = criterion(output, label)
            discriminator_real_acc.append(output.mean().item())

            noise = torch.randn(b_size, latent_dim,device=device)
            fake_images = generator(noise)
            label_fake = torch.zeros((b_size,), dtype=torch.float, device=device)
            output = discriminator(fake_images.detach()).view(-1)
            error_discriminator_fake = criterion(output, label_fake)
            discriminator_fake_acc.append(output.mean().item())
            error_discriminator = error_discriminator_real + error_discriminator_fake
            error_discriminator.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            label = torch.ones((b_size,), dtype=torch.float, device=device)
            output = discriminator(fake_images).view(-1)
            error_generator = criterion(output, label)
            error_generator.backward()
            D_G_z2 = output.mean().item()
            generator_optimizer.step()

            G_losses.append(error_generator.item())
            D_losses.append(error_discriminator.item())

        print(f"Epoch: {epoch}, discriminator fake error: {np.mean(discriminator_fake_acc):.3}, discriminator real acc: {np.mean(discriminator_real_acc):.3}")
        generator_scheduler.step()
        discriminator_scheduler.step()

    number = 1000
    test_images = get_test_images(test, number)
    input_images = get_train_images(number, test_set=test)
    evaluator = train_evaluator(trainloader, 5, class_amount, device, 3*32*32, 256)
    print("Correctly guessed ", eval_evaluator(trainloader, evaluator, device), "% of the dataset")

    gen_images = generate_random_images(generator, 32, number, device)
    visualize_images(input_images, gen_images, reconstruct=False)
    inv_gen_images = gen_images * 255
    inv_gen_images = inv_gen_images.type(torch.uint8)
    gen_distance = calculate_frechet_distance(get_distribution(test_images, evaluator, device).numpy(), get_distribution(gen_images, evaluator, device).numpy())
    print(f"Generation fid: {gen_distance}")
    visualize_images(input_images, inv_gen_images, reconstruct=False)

if __name__ == "__main__":
    main()
