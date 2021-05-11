import torch


def gradient_penalty(critic, labels, real, fake, device="cpu"):
      batch_size, channels, H, W = real.shape
      epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, H, W).to(device)
      interpolated_images = real * epsilon + fake * (1 - epsilon)

      # calculate critic scores
      mixed_scores = critic(interpolated_images, labels)

      # compute the gradient of the mixed scores with respect to the interpolated images.
      gradient = torch.autograd.grad(
          inputs=interpolated_images,
          outputs=mixed_scores,
          grad_outputs=torch.ones_like(mixed_scores),
          create_graph=True,
          retain_graph=True,
      )[0]  # we gonna get the first element of those

      gradient = gradient.view(gradient.shape[0], -1)  # flatten on all the other dimensions
      # Take L2 norm
      gradient_norm = gradient.norm(2, dim=1)
      gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
      return gradient_penalty