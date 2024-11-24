import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class VAEVisualizer:
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def visualize_samples(self, model, title, n_samples=64):
        with torch.no_grad():
            sample = model.sample(n_samples)
            sample = sample.cpu().view(n_samples, 1, 28, 28)

            plt.figure(figsize=(10, 10))
            for i in range(n_samples):
                plt.subplot(8, 8, i + 1)
                plt.imshow(sample[i][0], cmap='gray')
                plt.axis('off')
            plt.suptitle(title)
            plt.savefig(os.path.join(self.save_dir, f'{title.lower().replace(" ", "_")}.png'))
            plt.close()

    def visualize_reconstructions(self, model, data_loader, title):
        model.eval()
        with torch.no_grad():
            data, _ = next(iter(data_loader))
            recon, _, _ = model(data)

            comparison = torch.cat([
                data[:8].view(-1, 1, 28, 28),
                recon[:8].view(-1, 1, 28, 28)
            ])

            plt.figure(figsize=(12, 4))
            for i in range(16):
                plt.subplot(2, 8, i + 1)
                plt.imshow(comparison[i][0], cmap='gray')
                plt.axis('off')
            plt.suptitle(f'{title} - Original (top) vs Reconstruction (bottom)')
            plt.savefig(os.path.join(self.save_dir, f'{title.lower().replace(" ", "_")}_reconstruction.png'))
            plt.close()

    def plot_loss_comparison(self, bernoulli_losses, gaussian_losses):
        df = pd.DataFrame({
            'Epoch': range(1, len(bernoulli_losses) + 1),
            'Bernoulli VAE': bernoulli_losses,
            'Gaussian VAE': gaussian_losses
        })

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df.melt('Epoch', var_name='Model', value_name='Loss'),
                    x='Epoch', y='Loss', hue='Model')
        plt.title('Loss Comparison: Bernoulli VAE vs Gaussian VAE')
        plt.savefig(os.path.join(self.save_dir, 'loss_comparison.png'))
        plt.close()
