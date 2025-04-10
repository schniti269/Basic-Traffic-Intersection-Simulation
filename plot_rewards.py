import os
import re
import matplotlib.pyplot as plt

# Directory containing the saved models
SAVE_DIR = "saved_models"


def extract_data_from_filename(filename):
    """Extracts epoch and reward from the filename."""
    match = re.match(r"model_epoch_(\d+)_reward_(\d+)_.*", filename)
    if match:
        epoch = int(match.group(1))
        reward = int(match.group(2))
        return epoch, reward
    return None


def plot_rewards(save_dir):
    """Plots the reward over epochs from saved model filenames."""
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
                # Keep the reward for this epoch (overwrites if multiple files for the same epoch)
                epoch_rewards[epoch] = reward

    if not epoch_rewards:
        print("No valid model files found to plot.")
        return

    # Sort data by epoch
    sorted_epochs = sorted(epoch_rewards.keys())
    sorted_rewards = [epoch_rewards[epoch] for epoch in sorted_epochs]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_epochs, sorted_rewards, marker="o", linestyle="-")
    plt.title("Reward Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_plot.png")  # Save the plot
    print("Plot saved as reward_plot.png")
    # plt.show() # Uncomment to display the plot instead of saving


if __name__ == "__main__":
    plot_rewards(SAVE_DIR)
