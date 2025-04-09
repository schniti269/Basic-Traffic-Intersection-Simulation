import sys
import threading
import os
from tqdm import tqdm
import sys
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.utils import (
    logger,
    directionNumbers,
    waiting_times,
    noOfSignals,
    simulation,
    vehicles,
    stopLines,
    movingGap,
    defaultStop,
    PERFORMANCE_MODE,
    # Import assumed screen dimensions or define them here
    # Assuming based on max coordinates in shared/utils.py
    # screenWidth, # Uncomment if defined in utils
    # screenHeight, # Uncomment if defined in utils
)
from core.simulation.traffic_signal import initialize
from core.simulation.vehicle import generateVehicles
from core.agent.neural_model_01 import (
    NeuralTrafficController,
    get_vehicles_in_zones,
    DEFAULT_SCAN_ZONE_CONFIG,
)

# --- Screen Dimensions (Assumed) ---
# Define here if not imported from utils
screenWidth = 1400
screenHeight = 800
# --- End Screen Dimensions ---

# --- Training Configuration ---
MODEL_SAVE_DIR = "saved_models"
STEPS_PER_EPOCH = 10000
TOTAL_EPOCHS = 50
# --- End Configuration ---


def train_simulation():
    """Runs the traffic simulation training loop."""

    # --- GPU Check ---
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            logger.info(
                f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured."
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.error(f"GPU Memory Growth Error: {e}")
    else:
        logger.info("No GPU found, using CPU.")
    # --- End GPU Check ---

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger.info(f"Starting training for {TOTAL_EPOCHS} epochs...")

    # Initialize environment (needed for scan zone config access during training)
    # env = TrafficEnvironment()

    # Create neural network controller
    neural_controller = NeuralTrafficController(
        steps_per_epoch=STEPS_PER_EPOCH, total_epochs=TOTAL_EPOCHS
    )

    # Start background threads (simulation init, vehicle generation)
    thread1 = threading.Thread(name="initialization", target=initialize, args=())
    thread1.daemon = True
    thread1.start()
    logger.info("Initialization thread started")

    # Start vehicle generation thread
    thread2 = threading.Thread(
        name="generateVehicles", target=generateVehicles, args=()
    )
    thread2.daemon = True
    thread2.start()
    logger.info("Vehicle generation thread started")

    total_steps = 0
    running = True
    active_lights = [False] * noOfSignals  # Initial light state

    # Main simulation loop with tqdm progress bar
    total_simulation_steps = TOTAL_EPOCHS * STEPS_PER_EPOCH
    if PERFORMANCE_MODE:
        pbar = tqdm(
            total=total_simulation_steps,
            desc="Training Progress",
            unit="step",
            mininterval=1.0,
        )
    else:
        pbar = None

    while running and total_steps < total_simulation_steps:
        current_epoch = neural_controller.current_epoch

        # --- Calculate Step Metrics --- #
        waiting_vehicles_this_step = 0
        avg_speed = 0
        vehicle_count = 0
        emissions_this_step = 0

        for d in directionNumbers.values():
            for lane in range(3):
                for vehicle in vehicles[d][lane]:
                    if vehicle.waiting_time > 0:
                        waiting_vehicles_this_step += 1
                    avg_speed += vehicle.speed
                    vehicle_count += 1

        if vehicle_count > 0:
            avg_speed /= vehicle_count

        for vehicle in simulation:
            emissions_this_step += vehicle.update_emission()

        crashes_this_step = 0
        # --- End Metric Calculation --- #

        # --- Simulation Update (Movement & Cleanup) --- #
        vehicles_to_remove = []
        for vehicle in list(simulation):
            # Remove the initial skip check for crashed vehicles
            # if vehicle.crashed:
            #     continue
            # old_x, old_y = vehicle.x, vehicle.y  # Store pos before move
            # Initialize crashes_result for this vehicle iteration
            crashes_result = 0
            # Only call move if the vehicle is not already crashed from a previous iteration
            if not vehicle.crashed:
                crashes_result = vehicle.move(
                    vehicles, active_lights, stopLines, movingGap, simulation
                )
                crashes_this_step += (
                    crashes_result  # Accumulate crashes detected by move
                )

            # Off-screen check - Use defined dimensions
            off_screen = (
                (vehicle.direction == "right" and vehicle.x > screenWidth)
                or (vehicle.direction == "left" and vehicle.x + vehicle.width < 0)
                or (vehicle.direction == "down" and vehicle.y > screenHeight)
                or (vehicle.direction == "up" and vehicle.y + vehicle.height < 0)
            )
            # Check for removal *after* potential move/crash flag update
            # Update logic to remove if crashed OR off_screen
            if vehicle.crashed or off_screen:
                vehicles_to_remove.append(vehicle)

        for vehicle in vehicles_to_remove:
            simulation.remove(vehicle)
            try:
                vehicles[vehicle.direction][vehicle.lane].remove(vehicle)
                if vehicle.id in waiting_times[vehicle.direction]:
                    del waiting_times[vehicle.direction][vehicle.id]
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Failed to remove vehicle {getattr(vehicle, 'id', '?')} from dict/waiting_times: {e}"
                )
                pass
        # --- End Simulation Update --- #

        # --- Neural Controller Update --- #
        scan_zones = get_vehicles_in_zones(
            directionNumbers, vehicles, DEFAULT_SCAN_ZONE_CONFIG
        )
        # Controller update gives the action for the *next* step
        next_active_lights = neural_controller.update(
            scan_zones,
            avg_speed,
            crashes_this_step,
            waiting_vehicles_this_step,
            emissions_this_step,
            vehicle_count,
        )
        # --- End Neural Controller Update --- #

        # --- Training Epoch Check & Model Saving --- #
        if neural_controller.epoch_steps >= neural_controller.steps_per_epoch:
            # Save model *before* end_epoch resets metrics used for filename
            epoch_reward = neural_controller.epoch_accumulated_reward
            epoch_crashes = neural_controller.total_crashes_epoch
            save_filename = f"model_epoch_{current_epoch}_reward_{epoch_reward:.0f}_crashes_{epoch_crashes}.weights.h5"
            save_path = os.path.join(MODEL_SAVE_DIR, save_filename)
            neural_controller.save_model(save_path)

            # Now end the epoch (trains model, resets metrics)
            neural_controller.end_epoch()
        # --- End Training Epoch Check --- #

        # Use the action determined in the *previous* controller update for the *current* simulation step
        active_lights = next_active_lights

        # Update total steps and progress bar
        total_steps += 1
        if pbar:
            pbar.update(1)

    # --- End of Training Loop --- #
    if pbar:
        pbar.close()

    logger.info("Training finished.")

    # Save the final model
    final_save_filename = f"model_final_epoch_{neural_controller.current_epoch}_reward_{neural_controller.epoch_accumulated_reward:.0f}_crashes_{neural_controller.total_crashes_epoch}.weights.h5"
    final_save_path = os.path.join(MODEL_SAVE_DIR, final_save_filename)
    neural_controller.save_model(final_save_path)
    logger.info(f"Final model saved to {final_save_path}")


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
