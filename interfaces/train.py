import sys
import os
from tqdm import tqdm
import tensorflow as tf
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.utils import (
    logger,
    noOfSignals,
    PERFORMANCE_MODE,
    simulation,
)
# Importiere den neuen DQN-Agenten
from core.agent.neural_model_03 import TrafficDQNAgent
from shared.simulation_core import (
    reset_simulation,
    start_simulation_threads, 
    calculate_metrics, 
    update_simulation, 
    reset_agent, 
    wait_for_space
)

# --- Configuration --- #
MODEL_SAVE_DIR = "saved_models"
STEPS_PER_EPOCH = 5000  # Anzahl der Aktionen (nicht Ticks) pro Epoche
TOTAL_EPOCHS = 5000

# --- GPU Check und Konfiguration --- #
if not PERFORMANCE_MODE:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            logger.info(
                f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured."
            )
        except RuntimeError as e:
            logger.error(f"GPU Memory Growth Error: {e}")
    else:
        logger.info("No GPU found, using CPU.")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger.info(f"Starting training for {TOTAL_EPOCHS} epochs...")


def train_simulation():
    """Führt das Trainingsprogramm für die Verkehrssimulation aus."""
    # --- Initialisierung --- #
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # DQN Agent für die Ampelsteuerung
    dqn_agent = TrafficDQNAgent()
    dqn_agent.steps_per_epoch = STEPS_PER_EPOCH
    dqn_agent.total_epochs = TOTAL_EPOCHS

    # Starte Simulationsthreads und initialisiere Ampeln
    active_lights = start_simulation_threads()

    total_ticks = 0  # Zählt Simulationsticks (nicht Epoche-Aktionen)
    running = True
    
    # Gesamtzahl der Simulationsticks abschätzen (für Fortschrittsanzeige)
    est_ticks_per_epoch = STEPS_PER_EPOCH * dqn_agent.action_delay  # ~ Ticks pro Epoche
    total_simulation_ticks = TOTAL_EPOCHS * est_ticks_per_epoch
    
    # Fortschrittsanzeige
    pbar = tqdm(
        total=total_simulation_ticks,
        desc="Training Progress",
        unit="tick",
        mininterval=1.0,
        disable=not PERFORMANCE_MODE,
    )

    # --- Haupt-Trainingsschleife --- #
    logger.info("Starting main training loop with DQN controller...")

    while running and dqn_agent.current_epoch < TOTAL_EPOCHS:
        # --- Berechne aktuelle Metriken --- #
        metrics = calculate_metrics()

        # --- Simulationsschritt ausführen --- #
        update_results = update_simulation(active_lights)
        crashes_this_step = update_results["crashes_this_step"]
        newly_crossed_count = update_results["newly_crossed_count"]

        # --- Aktualisiere den DQN-Agent --- #
        next_active_lights = dqn_agent.update(
            simulation,
            metrics["avg_speed"],
            crashes_this_step,
            metrics["sum_sq_waiting_time"],
            metrics["emissions"],
            newly_crossed_count,
        )
        
        # --- Modell nach jeder Epoche speichern --- #
        # Prüfe, ob gerade eine Epoche abgeschlossen wurde
        if hasattr(dqn_agent, 'epoch_just_completed') and dqn_agent.epoch_just_completed:
            # Durchschnittliche Belohnung pro Aktion berechnen
            if dqn_agent.total_steps > 0:
                avg_reward = dqn_agent.epoch_accumulated_reward / max(1, dqn_agent.epoch_steps)
            else:
                avg_reward = 0
                
            epoch_crashes = dqn_agent.total_crashes_epoch
            avg_speed = dqn_agent.sum_avg_speeds_epoch / max(1, dqn_agent.vehicle_updates_epoch)
            
            # Dateiname mit informativen Metriken
            save_prefix = f"dqn_model_epoch_{dqn_agent.current_epoch}_reward_{avg_reward:.2f}_speed_{avg_speed:.2f}_crashes_{epoch_crashes}"
            save_path_prefix = os.path.join(MODEL_SAVE_DIR, save_prefix)
            dqn_agent.save_model(save_path_prefix)
            logger.info(f"Saved model checkpoint at epoch {dqn_agent.current_epoch} with avg reward: {avg_reward:.2f}")
        
        # Wende die Ampelzustände für den nächsten Schritt an
        active_lights = next_active_lights

        # Fortschritt aktualisieren
        total_ticks += 1
        pbar.update(1)
        
        # Auf Leertaste warten, falls nötig
        wait_for_space(dqn_agent)

    # --- Ende der Trainingsschleife --- #
    pbar.close()

    logger.info("Training finished.")

    # Speichere das finale Modell
    final_avg_reward = 0
    if dqn_agent.total_steps > 0:
        final_avg_reward = dqn_agent.epoch_accumulated_reward / max(1, dqn_agent.epoch_steps)
    
    final_save_prefix = f"dqn_model_epoch_{dqn_agent.current_epoch}_reward_{final_avg_reward:.2f}_crashes_{dqn_agent.total_crashes_epoch}"
    final_save_path_prefix = os.path.join(MODEL_SAVE_DIR, final_save_prefix)
    dqn_agent.save_model(final_save_path_prefix)
    logger.info(f"Final DQN model saved with prefix {final_save_path_prefix}")


# --- Hauptausführung --- #
if __name__ == "__main__":
    try:
        logger.info("Starting DQN training script...")
        train_simulation()
        logger.info("DQN training script finished.")
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
    finally:
        sys.exit()  # Ensure script exits
