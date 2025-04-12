import os
import re
import matplotlib.pyplot as plt
import argparse

# Directory containing the saved models
SAVE_DIR = "saved_models"


def extract_data_from_filename(filename):
    """Extracts epoch and reward from the filename."""
    match = re.match(r"model_epoch_(\d+)_reward_(-?\d+)_.*", filename)
    if match:
        epoch = int(match.group(1))
        reward = int(match.group(2))
        return epoch, reward
    return None


def plot_rewards(save_dir, max_epochs=None):
    """Plots the reward over epochs from saved model filenames.
    
    Args:
        save_dir (str): Directory containing model files
        max_epochs (int, optional): Maximum number of epochs to plot. If None, all epochs are plotted.
    """
    epoch_rewards = {}

    # List files and extract data
    try:
        filenames = os.listdir(save_dir)
    except FileNotFoundError:
        print(f"Error: Directory not found - {save_dir}")
        return

    for filename in filenames:
        # Process only one type of file per epoch to avoid duplicates
        if filename.endswith("_actor.weights.h5"):
            data = extract_data_from_filename(filename)
            if data:
                epoch, reward = data
                # Keep the highest reward for each epoch
                if epoch not in epoch_rewards or reward > epoch_rewards[epoch]:
                    epoch_rewards[epoch] = reward

    if not epoch_rewards:
        print("No valid model files found to plot.")
        return

    # Sort data by epoch
    sorted_epochs = sorted(epoch_rewards.keys())
    sorted_rewards = [epoch_rewards[epoch] for epoch in sorted_epochs]

    # Limit to max_epochs if specified
    if max_epochs is not None and max_epochs > 0:
        sorted_epochs = sorted_epochs[:max_epochs]
        sorted_rewards = sorted_rewards[:max_epochs]
        title_suffix = f" (First {max_epochs} Epochs)"
    else:
        title_suffix = ""

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_epochs, sorted_rewards, marker="o", linestyle="-")
    plt.title(f"Reward Over Epochs{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    
    # Save with custom filename if max_epochs is specified
    if max_epochs is not None and max_epochs > 0:
        filename = f"reward_plot_first_{max_epochs}_epochs.png"
    else:
        filename = "reward_plot.png"
    
    plt.savefig(filename)  # Save the plot
    print(f"Plot saved as {filename}")
    # plt.show() # Uncomment to display the plot instead of saving


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Plot rewards over epochs from saved models.')
    parser.add_argument('--max-epochs', type=int, default=None, 
                        help='Maximum number of epochs to plot (optional)')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR, 
                        help=f'Directory containing saved models (default: {SAVE_DIR})')
    
    args = parser.parse_args()
    plot_rewards(args.save_dir, args.max_epochs)
