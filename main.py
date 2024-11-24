import torch
from models.bernoulli_vae import BernoulliVAE
from models.gaussian_vae import GaussianVAE
from utils.data_loader import load_mnist
from utils.visualization import VAEVisualizer
import os
import time
from tqdm import tqdm

def train(model, train_loader, optimizer, device, model_name, epoch):
    model.train()
    train_loss = 0

    # Add progress bar for each epoch
    pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item()/len(data):.4f}'})

    return train_loss / len(train_loader.dataset)

def main():
    # Reduced epochs and increased batch size
    EPOCHS = 20  # Reduced from 50
    BATCH_SIZE = 256  # Increased from 128
    LEARNING_RATE = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    bernoulli_vae = BernoulliVAE().to(device)
    gaussian_vae = GaussianVAE().to(device)

    # Initialize optimizers
    b_optimizer = torch.optim.Adam(bernoulli_vae.parameters(), lr=LEARNING_RATE)
    g_optimizer = torch.optim.Adam(gaussian_vae.parameters(), lr=LEARNING_RATE)

    # Initialize visualization
    visualizer = VAEVisualizer()

    # Training loops
    bernoulli_losses = []
    gaussian_losses = []

    # Training Bernoulli VAE
    print("\nTraining Bernoulli VAE...")
    start_time = time.time()
    train_loader = load_mnist(BATCH_SIZE, is_bernoulli=True)

    for epoch in range(EPOCHS):
        loss = train(bernoulli_vae, train_loader, b_optimizer, device, "Bernoulli", epoch + 1)
        bernoulli_losses.append(loss)

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Time: {elapsed:.2f}s')

    bernoulli_time = time.time() - start_time
    print(f"\nBernoulli VAE training completed in {bernoulli_time:.2f} seconds")

    # Training Gaussian VAE
    print("\nTraining Gaussian VAE...")
    start_time = time.time()
    train_loader = load_mnist(BATCH_SIZE, is_bernoulli=False)

    for epoch in range(EPOCHS):
        loss = train(gaussian_vae, train_loader, g_optimizer, device, "Gaussian", epoch + 1)
        gaussian_losses.append(loss)

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Time: {elapsed:.2f}s')

    gaussian_time = time.time() - start_time
    print(f"\nGaussian VAE training completed in {gaussian_time:.2f} seconds")

    # Visualization
    print("\nGenerating visualizations...")
    visualizer.visualize_samples(bernoulli_vae, "Bernoulli VAE Samples")
    visualizer.visualize_samples(gaussian_vae, "Gaussian VAE Samples")
    visualizer.visualize_reconstructions(bernoulli_vae, train_loader, "Bernoulli VAE")
    visualizer.visualize_reconstructions(gaussian_vae, train_loader, "Gaussian VAE")
    visualizer.plot_loss_comparison(bernoulli_losses, gaussian_losses)

    print(f"\nFinal Gaussian VAE sigma value: {torch.exp(gaussian_vae.log_sigma).item():.4f}")
    print(f"\nTotal training time: {bernoulli_time + gaussian_time:.2f} seconds")

if __name__ == "__main__":
    main()
