import os
import re
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime

# Directory containing the saved models
SAVE_DIR = "saved_models"


def extract_data_from_filename(filename):
    """Extracts epoch and reward from the filename."""
    # Pattern für PPO-Modelle
    ppo_match = re.match(r"model_epoch_(\d+)_reward_(-?\d+)_.*", filename)
    if ppo_match:
        epoch = int(ppo_match.group(1))
        reward = int(ppo_match.group(2))
        return epoch, reward, "ppo"
        
    # Pattern für DQN-Modelle mit dem Format "dqn_model_epoch_20_avg_reward_1144.01_speed_2.50_crashes_0.weights.h5"
    dqn_match_avg = re.match(r"dqn_model_epoch_(\d+)_avg_reward_(-?\d+\.?\d*)_.*", filename)
    if dqn_match_avg:
        epoch = int(dqn_match_avg.group(1))
        try:
            reward = float(dqn_match_avg.group(2))
        except ValueError:
            reward = 0
        return epoch, reward, "dqn"
        
    # Alternatives Pattern für DQN-Modelle mit einfachem "reward_" Format
    dqn_match = re.match(r"dqn_model_epoch_(\d+)_reward_(-?\d+\.?\d*)_.*", filename)
    if dqn_match:
        epoch = int(dqn_match.group(1))
        try:
            reward = float(dqn_match.group(2))
        except ValueError:
            reward = 0
        return epoch, reward, "dqn"
        
    return None


def get_file_timestamp(filepath):
    """Gibt den Zeitstempel der Datei zurück."""
    return os.path.getmtime(filepath)


def plot_rewards(save_dir, max_epochs=None, smooth_window=None, model_type=None, use_indices=False):
    """Plots the reward over epochs from saved model filenames.
    
    Args:
        save_dir (str): Directory containing model files
        max_epochs (int, optional): Maximum number of epochs to plot. If None, all epochs are plotted.
        smooth_window (int, optional): Window size for smoothing the reward curve. If None, no smoothing is applied.
        model_type (str, optional): Filter model types ('ppo', 'simple', or 'dqn'). If None, all models are included.
        use_indices (bool, optional): Use indices instead of epoch numbers for x-axis.
    """
    # Liste aller Modelldateien mit Metadaten: (Dateiname, Zeitstempel, Epoche, Belohnung, Modelltyp)
    model_files = []

    # List files and extract data
    try:
        filenames = os.listdir(save_dir)
    except FileNotFoundError:
        print(f"Error: Directory not found - {save_dir}")
        return

    for filename in filenames:
        # Überspringe Dateien, die keine Modelle sind
        if not filename.endswith(".weights.h5"):
            continue
            
        data = extract_data_from_filename(filename)
        if data:
            epoch, reward, detected_type = data
            
            # Filtere nach Modelltyp, falls angegeben
            if model_type is not None and model_type != detected_type:
                continue
                
            # Füge Datei mit Zeitstempel hinzu
            filepath = os.path.join(save_dir, filename)
            timestamp = get_file_timestamp(filepath)
            model_files.append((filename, timestamp, epoch, reward, detected_type))

    if not model_files:
        print("No valid model files found to plot.")
        return

    # Sortiere nach Zeitstempel (wann die Datei erstellt wurde)
    model_files.sort(key=lambda x: x[1])
    
    # Extrahiere sortierte Epochen und Belohnungen
    if use_indices:
        # Verwende Index als X-Achse (fortlaufende Nummer)
        sorted_epochs = list(range(len(model_files)))
        x_label = "Model Index (Time-ordered)"
    else:
        # Verwende Epochennummern, behalte Duplikate
        sorted_epochs = [file[2] for file in model_files]
        x_label = "Epoch"
        
    sorted_rewards = [file[3] for file in model_files]
    sorted_timestamps = [datetime.fromtimestamp(file[1]).strftime('%H:%M:%S') for file in model_files]

    # Begrenze auf max_epochs falls angegeben
    if max_epochs is not None and max_epochs > 0 and len(sorted_epochs) > max_epochs:
        sorted_epochs = sorted_epochs[:max_epochs]
        sorted_rewards = sorted_rewards[:max_epochs]
        sorted_timestamps = sorted_timestamps[:max_epochs]
        title_suffix = f" (First {max_epochs} Models)"
    else:
        title_suffix = f" ({len(sorted_epochs)} Models)"
        
    # Anwenden von Glättung, falls gewünscht
    if smooth_window and smooth_window > 1 and len(sorted_rewards) > smooth_window:
        smoothed_rewards = []
        for i in range(len(sorted_rewards)):
            start = max(0, i - smooth_window // 2)
            end = min(len(sorted_rewards), i + smooth_window // 2 + 1)
            smoothed_rewards.append(np.mean(sorted_rewards[start:end]))
    else:
        smoothed_rewards = sorted_rewards

    # Plotting
    plt.figure(figsize=(14, 8))
    
    # Originaldaten mit geringer Transparenz
    plt.plot(sorted_epochs, sorted_rewards, 'o', alpha=0.3, markersize=3, label="Original Data")
    
    # Geglättete Linie (oder Originallinie, falls keine Glättung angewendet wurde)
    plt.plot(sorted_epochs, smoothed_rewards, '-', linewidth=2, label="Smoothed" if smooth_window else "Trend")
    
    # Grafik-Formatierung
    model_type_str = f" ({model_type.upper()} Models)" if model_type else ""
    plt.title(f"Reward Over Time{title_suffix}{model_type_str}", fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Statistik hinzufügen
    if len(sorted_rewards) > 0:
        avg_reward = np.mean(sorted_rewards)
        max_reward = np.max(sorted_rewards)
        min_reward = np.min(sorted_rewards)
        stats_text = f"Avg: {avg_reward:.1f}\nMax: {max_reward:.1f}\nMin: {min_reward:.1f}"
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Speichere mit angepasstem Dateinamen
    filename_parts = []
    if model_type:
        filename_parts.append(model_type)
    filename_parts.append("reward_plot")
    if use_indices:
        filename_parts.append("by_index")
    if max_epochs is not None and max_epochs > 0:
        filename_parts.append(f"first_{max_epochs}")
    if smooth_window:
        filename_parts.append(f"smooth_{smooth_window}")
    
    filename = "_".join(filename_parts) + ".png"
    
    plt.savefig(filename, dpi=300)  # Höhere Auflösung
    print(f"Plot saved as {filename}")


if __name__ == "__main__":
    # Befehlszeilenargumente einrichten
    parser = argparse.ArgumentParser(description='Plot rewards over epochs from saved models.')
    parser.add_argument('--max-epochs', type=int, default=None, 
                        help='Maximum number of epochs to plot (optional)')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR, 
                        help=f'Directory containing saved models (default: {SAVE_DIR})')
    parser.add_argument('--smooth', type=int, default=None,
                        help='Window size for smoothing the reward curve (optional)')
    parser.add_argument('--model-type', type=str, choices=['ppo', 'simple', 'dqn'], default=None,
                        help='Filter model types (ppo, simple, or dqn)')
    parser.add_argument('--use-indices', action='store_true',
                        help='Use indices instead of epoch numbers for x-axis (useful for models with same epoch)')
    
    args = parser.parse_args()
    plot_rewards(args.save_dir, args.max_epochs, args.smooth, args.model_type, args.use_indices)
