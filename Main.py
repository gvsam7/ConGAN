"""
Author: Georgios Voulgaris
Date: 07/05/2021
Description: Conditional WGAN_GP architecture. The aim of this project is to generate synthetic images and see how they
affect class imbalanced dataset and Deep architecture performance in the absence of data.
"""

import torch
import torch.optim as optim
import torchvision
from torch.utils import tensorboard
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torch.utils.data import DataLoader
import argparse
from models.Discriminator import Discriminator
from models.Generator import Generator
from models.InitialiseWeights import initialise_weights
from utils.GradientPenalty import gradient_penalty
from utils.SaveLoadChkpnt import save_checkpoint, load_checkpoint
import wandb
wandb.tensorboard.patch(root_logdir='logs/GAN_MNIST/')
wandb.init(project="GAN", sync_tensorboard=True)


# Hyperparameters
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--channels_img", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--features-critic", type=int, default=16)
    parser.add_argument("--features-gen", type=int, default=16)
    parser.add_argument("--critic-iterations", type=int, default=5)
    parser.add_argument("--lambda_gp", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--gen-embedding", type=int, default=100)
    return parser.parse_args()


def main():
    args = arguments()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    transforms = Compose(
        [
            Resize(args.img_size),
            ToTensor(),
            Normalize(
                [0.5 for _ in range(args.channels_img)], [0.5 for _ in range(args.channels_img)]),
        ]
    )

    dataset = datasets.FashionMNIST(root="dataset/", transform=transforms, download=True)
    # comment mnist above and uncomment below for training on CelebA
    # dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(args.z_dim, args.channels_img, args.features_gen,
                    args.num_classes, args.img_size, args.gen_embedding).to(device)
    critic = Discriminator(args.channels_img, args.features_critic,
                           args.num_classes, args.img_size).to(device)
    initialise_weights(gen)
    initialise_weights(critic)

    wandb.watch(gen)
    wandb.watch(critic)

    # initialisate optimiser
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(32, args.z_dim, 1, 1).to(device)
    writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
    writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
    step = 0

    gen.train()
    critic.train()

    for epoch in range(args.epochs):
        for batch_idx, (real, labels) in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]
            labels = labels.to(device)

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(args.critic_iterations):
                noise = torch.randn(cur_batch_size, args.z_dim, 1, 1).to(device)
                fake = gen(noise, labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake, labels).reshape(-1)
                gp = gradient_penalty(critic, labels, real, fake, device=device)
                loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake)) + args.lambda_gp * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # wandb
            train_steps = len(loader) * (epoch + 1)
            wandb.log({"Loss D": loss_critic, "Loss G:": loss_gen})

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] Batch {batch_idx}/{len(loader)} \
                                Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(noise, labels)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1

                wandb.log(
                    {
                        "Real": [wandb.Image(i) for i in img_grid_real],
                        "Fake": [wandb.Image(i) for i in img_grid_fake],
                           }
                )


if __name__ == "__main__":
    main()
