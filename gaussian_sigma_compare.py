import torch
from models.gaussian_vae import GaussianVAE
from utils.data_loader import load_mnist
from utils.visualization import VAEVisualizer
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class GaussianVAEWithSigma(GaussianVAE):
    def __init__(self, fixed_sigma=None):
        super().__init__()
        if fixed_sigma is not None:
            # Use a fixed sigma instead of learning it
            self.register_buffer('fixed_sigma', torch.tensor([fixed_sigma]))
            self.use_fixed_sigma = True
        else:
            self.use_fixed_sigma = False

    def loss_function(self, recon_x, x, mu, logvar):
        sigma = self.fixed_sigma if self.use_fixed_sigma else torch.exp(self.log_sigma)
        MSE = 0.5 * torch.sum((recon_x - x.view(-1, self.input_dim)).pow(2)) / (sigma**2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        log_sigma_term = self.input_dim * torch.log(sigma)
        return MSE + KLD + log_sigma_term, MSE, KLD

def train_model(model, train_loader, sigma_value, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in tqdm(range(epochs), desc=f'Training σ={sigma_value}'):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, _, _ = model.loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        losses.append(epoch_loss / len(train_loader.dataset))

    return model, losses

def compare_sigmas():
    # Try different sigma values
    sigma_values = [0.1, 0.5, 1.0, 2.0]
    train_loader = load_mnist(batch_size=256, is_bernoulli=False)
    visualizer = VAEVisualizer(save_dir='results_sigma_comparison')

    # Store results for each sigma
    all_models = {}
    all_losses = {}

    # Train models with different sigmas
    for sigma in sigma_values:
        print(f"\nTraining model with σ={sigma}")
        model = GaussianVAEWithSigma(fixed_sigma=sigma)
        trained_model, losses = train_model(model, train_loader, sigma)
        all_models[sigma] = trained_model
        all_losses[sigma] = losses

        # Generate and save samples
        visualizer.visualize_samples(
            trained_model,
            f"Gaussian VAE Samples (σ={sigma})"
        )
        visualizer.visualize_reconstructions(
            trained_model,
            train_loader,
            f"Gaussian VAE (σ={sigma})"
        )

    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    for sigma, losses in all_losses.items():
        plt.plot(losses, label=f'σ={sigma}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different σ Values')
    plt.legend()
    plt.savefig('results_sigma_comparison/loss_comparison_sigmas.png')
    plt.close()

if __name__ == "__main__":
    compare_sigmas()
