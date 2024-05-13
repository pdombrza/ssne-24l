import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from evaluator import train_evaluator, calculate_frechet_distance, get_distribution, eval_evaluator

class Discriminator(nn.Module):
    def __init__(self, inputsList: list[int], neuronsList: list[int], outputs: int, activationFunc: nn.Module = nn.ReLU()):
        super(Discriminator, self).__init__()
        self._layers = nn.Sequential()
        for inputs, neurons in zip(inputsList, neuronsList):
            self._layers.append(nn.Linear(inputs, neurons))
            # self._layers.append(nn.BatchNorm1d(neurons))
            self._layers.append(activationFunc)
        self._layers.append(nn.Linear(neuronsList[-1], outputs))

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self._layers(x)

class Generator(nn.Module):
    def __init__(self, inputsList: list[int], neuronsList: list[int], outputs: int, activationFunc: nn.Module = nn.ReLU()):
        super(Generator, self).__init__()
        self._layers = nn.Sequential()
        for inputs, neurons in zip(inputsList, neuronsList):
            self._layers.append(nn.Linear(inputs, neurons))
            self._layers.append(nn.BatchNorm1d(neurons))
            self._layers.append(activationFunc)
        self._layers.append(nn.Linear(neuronsList[-1], outputs))
        
    def forward(self, x):
        x_hat = torch.sigmoid(self._layers(x))
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

def generate_random_images(model, latent_dim, n_imgs, device):
    model.eval()
    with torch.no_grad():
        generated_imgs = model(torch.randn(n_imgs, latent_dim, device=device))
    generated_imgs = generated_imgs.cpu()
    return generated_imgs

def main():
    device = torch.device("cuda")
    prepare_cuda()
    data_path = "./trafic_32"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_size = 256
    dataset = ImageFolder(data_path, transform=transform)
    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=8)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)
    class_amount = len(dataset.classes)

    latent_dim = 32
    generator = Generator([32, 128, 128, 256, 512], [128, 128, 256, 512, 1024], 3072, nn.ReLU()).to(device)
    discriminator = Discriminator([3072, 256, 128, 64], [256, 128, 64, 32], 1, nn.ReLU()).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=generator_optimizer, gamma=0.99)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=discriminator_optimizer, gamma=0.99)

    criterion = nn.MSELoss()

    G_losses = []
    D_losses = []
    num_epochs = 650
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

            noise = torch.randn(b_size, latent_dim, device=device)
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
    torch.save(inv_gen_images.cpu().detach(), "piatek_Dombrzalski_Kiełbus.pt")

if __name__ == "__main__":
    main()
