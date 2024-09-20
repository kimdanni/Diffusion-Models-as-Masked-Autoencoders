from diffusers import DDPMScheduler
import torch


class CustomDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        # Initialize the base class first
        super().__init__(*args, **kwargs)
        self._adjust_betas(0.8)  # Adjust the betas by multiplying them by 0.8

    def _adjust_betas(self, factor: float = 0.8):
        """
        Adjust beta values by a specified factor after initialization and recompute alphas and related variables.
        """
        # Scale the betas by the specified factor (e.g., 0.8)
        self.betas = self.betas ** factor

        # Recompute alphas and alphas_cumprod based on new betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Optionally print or log the updated values for verification
        print(f"Betas adjusted by factor {factor} and recalculated.")
        
    def noise_sampling(self, x):
        bs, c, h, w = x.shape
        noise = torch.randn(x.shape).to(x.device)
        timesteps = torch.randint(0, self.config.num_train_timesteps, (bs,), device=x.device).long()
        samples = self.add_noise(x, noise, timesteps)
        return samples, timesteps

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    scheduler = CustomDDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    x = torch.zeros(10, 3, 224, 224)  # Create a tensor filled with zeros
    x[:, 1, :, :] = 1  # Set the green channel
    x, timesteps = scheduler.noise_sampling(x)
    for i in range(x.shape[0]):
        plt.imshow(x[i].permute(1, 2, 0).cpu().numpy())
        plt.show()
        print(timesteps[i])