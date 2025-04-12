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
    simulation,  # Wichtig: Füge simulation hier hinzu
)
from core.agent.neural_model_01 import (
    NeuralTrafficControllerPPO,
    UPDATE_EVERY_N_STEPS,
)
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
STEPS_PER_EPOCH = 6000
TOTAL_EPOCHS = 5000

# --- Simulation Environment Setup --- #
if not PERFORMANCE_MODE:
    # --- GPU Check --- #
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
    # --- End GPU Check --- #

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger.info(f"Starting training for {TOTAL_EPOCHS} epochs...")


def train_simulation():
    """Runs the traffic simulation training loop."""
    # --- Initialization ---
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # PPO Controller
    neural_controller = NeuralTrafficControllerPPO(
        steps_per_epoch=STEPS_PER_EPOCH, total_epochs=TOTAL_EPOCHS
    )

    # Starte Simulationsthreads und initialisiere Ampeln auf rot
    active_lights = start_simulation_threads()

    total_steps = 0
    running = True

    total_simulation_steps = TOTAL_EPOCHS * STEPS_PER_EPOCH
    pbar = tqdm(
        total=total_simulation_steps,
        desc="Training Progress",
        unit="step",
        mininterval=1.0,
        disable=not PERFORMANCE_MODE,  # Disable tqdm if not in performance mode
    )

    # --- Main Training Loop --- #
    logger.info("Starting main training loop...")

    while running and total_steps < total_simulation_steps:
        current_epoch = neural_controller.current_epoch

        # --- Berechne Metriken vor der Bewegung --- #
        metrics = calculate_metrics()

        # --- Simulationsschritt ausführen --- #
        update_results = update_simulation(active_lights)
        crashes_this_step = update_results["crashes_this_step"]
        newly_crossed_count = update_results["newly_crossed_count"]

        # --- Neural Controller Update --- #
        next_active_lights = neural_controller.update(
            simulation,  # Globale Simulation gruppe
            metrics["avg_speed"],
            crashes_this_step,
            metrics["sum_sq_waiting_time"],
            metrics["emissions"],
            newly_crossed_count,
        )
        
        # --- Training Epoch Check & Model Saving --- #
        if neural_controller.epoch_steps >= neural_controller.steps_per_epoch:
            # Save model *before* end_epoch resets metrics used for filename
            epoch_reward = neural_controller.epoch_accumulated_reward
            epoch_crashes = neural_controller.total_crashes_epoch
            # Use a prefix for saving actor/critic models
            save_prefix = f"model_epoch_{current_epoch}_reward_{epoch_reward:.0f}_crashes_{epoch_crashes}"
            save_path_prefix = os.path.join(MODEL_SAVE_DIR, save_prefix)
            neural_controller.save_model(save_path_prefix)

            # Now end the epoch (resets metrics, PPO training happens in update)
            neural_controller.end_epoch()
            
            # Reset the simulation environment at the end of each epoch
            reset_simulation()
            
            # Reset neural controller's internal state for clean start at each epoch
            neural_controller = reset_agent(neural_controller)
            
            logger.info(
                f"Epoch {current_epoch} finished and environment fully reset."
            )
        
        # Apply the determined action for the next simulation step
        active_lights = next_active_lights

        total_steps += 1
        pbar.update(1)

        # Warte auf Leertaste falls nötig
        wait_for_space(neural_controller)

    # --- End of Training Loop --- #
    pbar.close()

    logger.info("Training finished.")

    # Save the final model (using prefix)
    final_save_prefix = f"model_final_epoch_{neural_controller.current_epoch}_reward_{neural_controller.epoch_accumulated_reward:.0f}_crashes_{neural_controller.total_crashes_epoch}"
    final_save_path_prefix = os.path.join(MODEL_SAVE_DIR, final_save_prefix)
    neural_controller.save_model(final_save_path_prefix)
    logger.info(f"Final PPO models saved with prefix {final_save_path_prefix}")


# --- Main Execution --- #
if __name__ == "__main__":
    try:
        logger.info("Starting training script...")
        train_simulation()
        logger.info("Training script finished.")
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
    finally:
        sys.exit()  # Ensure script exits
