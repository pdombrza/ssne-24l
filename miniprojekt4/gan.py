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
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out  = nn.Linear(hidden_dim, 1)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.LeakyReLU(self.fc_1(x))
        x = self.LeakyReLU(self.fc_2(x))
        x = self.fc_out(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc_1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.fc_1(x))
        h     = self.LeakyReLU(self.fc_2(h))
        
        x_hat = torch.sigmoid(self.fc_3(h))
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

# def generate_reconstructions(model, input_images, device):
#     model.eval()
#     with torch.no_grad():
#         reconst_imgs = model(input_images.to(device))
#     reconst_imgs = reconst_imgs.cpu()
#     return reconst_imgs

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


def generate_random_images(model, latent_dim,  n_imgs, device):
    # Generate images
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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # load data
    batch_size = 256
    dataset = ImageFolder(data_path, transform=transform)
    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)
    class_amount = len(dataset.classes)

    # Models
    latent_dim = 32
    generator = Generator(latent_dim=latent_dim, hidden_dim=256, output_dim=3072).to(device)
    discriminator = Discriminator(hidden_dim=256, input_dim=3072).to(device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    generator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=generator_optimizer, gamma=0.99)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=discriminator_optimizer, gamma=0.99)

    # loss
    criterion = nn.MSELoss()

    fixed_noise = torch.randn(16, latent_dim,device=device)

    G_losses = []
    D_losses = []
    num_epochs = 10
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        discriminator_fake_acc = []
        discriminator_real_acc = []
        for i, data in enumerate(trainloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator_optimizer.zero_grad()
            # Format batch
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            label = torch.ones((b_size,), dtype=torch.float, device=device) # Setting labels for real images
            # Forward pass real batch through D
            output = discriminator(real_images).view(-1)
            # Calculate loss on all-real batch
            error_discriminator_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            discriminator_real_acc.append(output.mean().item())

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_dim,device=device)
            # Generate fake image batch with Generator
            fake_images = generator(noise)
            label_fake = torch.zeros((b_size,), dtype=torch.float, device=device)
            # Classify all fake batch with Discriminator
            output = discriminator(fake_images.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            error_discriminator_fake = criterion(output, label_fake)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            discriminator_fake_acc.append(output.mean().item())
            # Compute error of D as sum over the fake and the real batches
            error_discriminator = error_discriminator_real + error_discriminator_fake
            error_discriminator.backward()
            # Update D
            discriminator_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator_optimizer.zero_grad()
            label = torch.ones((b_size,), dtype=torch.float, device=device)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake_images).view(-1)
            # Calculate G's loss based on this output
            error_generator = criterion(output, label)
            # Calculate gradients for G
            error_generator.backward()
            D_G_z2 = output.mean().item()
            # Update G
            generator_optimizer.step()

            # Output training stats
            # Save Losses for plotting later
            G_losses.append(error_generator.item())
            D_losses.append(error_discriminator.item())

        print(f"Epoch: {epoch}, discriminator fake error: {np.mean(discriminator_fake_acc):.3}, discriminator real acc: {np.mean(discriminator_real_acc):.3}")
        generator_scheduler.step()
        discriminator_scheduler.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            grid = torchvision.utils.make_grid(fake)
            grid = grid.permute(1, 2, 0)
            plt.figure(figsize=(10,10))
            plt.title(f"Generations")
            plt.imshow(grid)
            plt.axis('off')
            plt.show()
    number = 1000
    test_images = get_test_images(test, number)

    # Reconstruct
    # input_images = get_train_images(number, test_set=test)
    evaluator = train_evaluator(trainloader, 5, class_amount, device, 3*32*32, 256)
    print("Correctly guessed ", eval_evaluator(trainloader, evaluator, device), "% of the dataset")
    # reconst_images = generate_reconstructions(generator, input_images, device)
    # visualize_images(input_images, reconst_images)
    # reconst_distance = calculate_frechet_distance(get_distribution(test_images, evaluator, device).numpy(), get_distribution(reconst_images, evaluator, device).numpy())
    # print(f"Reconstruction fid: {reconst_distance}")

    # Generate
    gen_images = generate_random_images(generator, 32, number, device)
    print(gen_images.size())
    # visualize_images(input_images, gen_images, reconstruct=False)
    gen_distance = calculate_frechet_distance(get_distribution(test_images, evaluator, device).numpy(), get_distribution(gen_images, evaluator, device).numpy())
    print(f"Generation fid: {gen_distance}")
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 2, 2, 2 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    inv_gen_images = invTrans(gen_images)
    torch.save(inv_gen_images.cpu().detach(),"piatek_Dombrzalski_Kiełbus.pt")

if __name__ == "__main__":
    main()