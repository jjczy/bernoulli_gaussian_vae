import numpy as np
import matplotlib.pyplot as plt

def visualize_loss_functions():
    # Setup the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Binary Cross-Entropy Loss
    pred_probs = np.linspace(0.01, 0.99, 100)
    bce_loss_true1 = -np.log(pred_probs)
    bce_loss_true0 = -np.log(1 - pred_probs)

    ax1.plot(pred_probs, bce_loss_true1, label='BCE Loss (True=1)', color='blue')
    ax1.plot(pred_probs, bce_loss_true0, label='BCE Loss (True=0)', color='green')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Loss')
    ax1.set_title('Binary Cross-Entropy Loss')
    ax1.grid(True)
    ax1.legend()

    # MSE Loss with different sigmas
    x = np.linspace(-2, 2, 100)
    sigma_values = [0.5, 1.0, 2.0]
    colors = ['blue', 'green', 'red']

    for sigma, color in zip(sigma_values, colors):
        mse_loss = (x**2)/(2*sigma**2) + np.log(sigma)
        ax2.plot(x, mse_loss, label=f'MSE (σ={sigma})', color=color)

    ax2.set_xlabel('Prediction Error (x - x̂)')
    ax2.set_ylabel('Loss')
    ax2.set_title('MSE Loss with Different σ Values')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('loss_functions_comparison.png')
    plt.close()

if __name__ == "__main__":
    visualize_loss_functions()
