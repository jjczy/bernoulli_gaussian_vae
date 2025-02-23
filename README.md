# bernoulli_gaussian_vae
# VAE Comparison: Bernoulli vs Gaussian

This project implements and compares two types of Variational Autoencoders (VAEs) on the MNIST dataset: Bernoulli VAE and Gaussian VAE. It explores their different behaviors, loss functions, and generation capabilities.

## Project Structure
```
vae_comparison/
├── models/
│   ├── __init__.py
│   ├── base_vae.py       # Base VAE architecture
│   ├── bernoulli_vae.py  # Bernoulli VAE implementation
│   └── gaussian_vae.py   # Gaussian VAE implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # MNIST data loading utilities
│   └── visualization.py  # Visualization tools
├── main.py               # Main training script
├── gaussian_sigma_comparison.py  # Script for sigma analysis
├── loss_visualizer.py    # Loss function visualization
└── requirements.txt      # Project dependencies
```

## Setup and Installation

1. Create a virtual environment (Python 3.10 recommended):
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

### Main Training
To train both VAE models and generate comparisons:
```bash
python main.py
```
This will:
- Train both Bernoulli and Gaussian VAEs
- Generate sample images
- Create reconstruction visualizations
- Plot loss comparisons

### Sigma Comparison
To analyze different sigma values for the Gaussian VAE:
```bash
python gaussian_sigma_comparison.py
```

### Loss Function Visualization
To visualize the different loss functions:
```bash
python loss_visualizer.py
```

## Model Details

### Bernoulli VAE
- Uses Binary Cross-Entropy Loss
- Suitable for binary data (like MNIST)
- Loss function: BCE + KL Divergence
- Produces sharper outputs

### Gaussian VAE
- Uses Mean Squared Error Loss with learned variance
- Suitable for continuous data
- Loss function: MSE + KL Divergence + log-variance term
- Outputs can be more nuanced but potentially blurrier

## Results

The project generates several visualization files:

1. `bernoulli_vae_samples.png`: Generated samples from Bernoulli VAE
2. `gaussian_vae_samples.png`: Generated samples from Gaussian VAE
3. `bernoulli_vae_reconstruction.png`: Original vs reconstructed images (Bernoulli)
4. `gaussian_vae_reconstruction.png`: Original vs reconstructed images (Gaussian)
5. `loss_comparison.png`: Training loss comparison between models
6. `loss_functions_comparison.png`: Visualization of loss functions

### Output Directory Structure
```
results/
├── bernoulli_vae_samples.png
├── gaussian_vae_samples.png
├── bernoulli_vae_reconstruction.png
├── gaussian_vae_reconstruction.png
└── loss_comparison.png
```

## Key Observations

1. Bernoulli VAE:
   - Better suited for MNIST due to binary nature of data
   - Produces sharper, more defined digits
   - Higher initial loss but good convergence

2. Gaussian VAE:
   - More flexible for continuous values
   - Can produce blurrier outputs
   - Lower loss values but may underfit binary data

## Model Architecture
Both VAEs share the same basic architecture:
- Encoder: 784 → 400 → 400 → 40 (latent)
- Decoder: 40 → 400 → 400 → 784
- Latent dimension: 40
- Hidden layers: 400 units each

## Hyperparameters
- Batch size: 256
- Learning rate: 0.001
- Epochs: 20
- Optimizer: Adam
- Input dimension: 784 (28×28 images)
- Hidden dimension: 400
- Latent dimension: 40

## Contributing
Feel free to open issues or submit pull requests for improvements. Some areas for potential enhancement:
- Adding more VAE variants
- Implementing additional datasets
- Improving visualization capabilities
- Optimizing training performance

## Requirements
- Python 3.10 or higher
- PyTorch
- torchvision
- matplotlib
- numpy
- pandas
- seaborn

## References
- [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## Author
Jessica Zhou
