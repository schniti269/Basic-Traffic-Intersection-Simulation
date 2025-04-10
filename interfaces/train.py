import sys
import threading
import os
from tqdm import tqdm
import tensorflow as tf
import time

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
    # screenWidth, # Assuming these are defined below for now
    # screenHeight, # Assuming these are defined below for now
    # SpatialGrid,  # REMOVED Import
)
from core.simulation.traffic_signal import initialize
from core.simulation.vehicle import generateVehicles
from core.agent.neural_model_01 import (
    # NeuralTrafficController, # Old REINFORCE class
    NeuralTrafficControllerPPO,  # New PPO class
    # get_vehicles_in_zones, # REMOVED
    # DEFAULT_SCAN_ZONE_CONFIG, # REMOVED
)

# --- Configuration --- #
MODEL_SAVE_DIR = "saved_models"
STEPS_PER_EPOCH = 2500
TOTAL_EPOCHS = 5000
SCREEN_WIDTH = 1400  # Define screen dimensions clearly
SCREEN_HEIGHT = 800
# GRID_CELL_SIZE = ( # REMOVED Constant
#     100  # Size of grid cells for collision detection (tune based on vehicle sizes)
# )

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
    global currentGreen, currentYellow  # Allow modification

    # --- Initialization ---
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # PPO Controller
    neural_controller = NeuralTrafficControllerPPO(
        steps_per_epoch=STEPS_PER_EPOCH, total_epochs=TOTAL_EPOCHS
    )

    # Initialize Traffic Signals
    initialize()
    active_lights = [False] * noOfSignals  # Start with all red

    # Initialize Spatial Grid # REMOVED
    # spatial_grid = SpatialGrid(SCREEN_WIDTH, SCREEN_HEIGHT, GRID_CELL_SIZE)

    # Start Vehicle Generation Thread
    thread = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    thread.daemon = True
    thread.start()
    logger.info("Vehicle generation thread started")

    # Start background threads
    thread1 = threading.Thread(name="initialization", target=initialize, args=())
    thread1.daemon = True
    thread1.start()
    logger.info("Initialization thread started")

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

        # --- Clear and Populate Spatial Grid --- # REMOVED
        # spatial_grid.clear()
        # for vehicle in simulation:
        #     spatial_grid.insert(vehicle)
        # --- End Grid Update --- # REMOVED

        # --- Metric Calculation --- #
        waiting_vehicles_this_step = 0
        sum_speed = 0
        vehicle_count = 0
        emissions_this_step = 0
        sum_sq_waiting_time = 0  # New metric
        vehicles_crossed_this_step = 0  # New metric (will be calculated post-move)

        # @cython.compile / @numba.jit (Potential Optimization Target for this loop)
        # Single loop over the simulation group to gather metrics PRE-MOVE
        for vehicle in simulation:
            # Check waiting status
            if not vehicle.crashed and vehicle.waiting_time > 0:
                waiting_vehicles_this_step += 1

            # Sum speed for average calculation
            sum_speed += vehicle.speed
            vehicle_count += 1

            # Calculate instantaneous emission for this step using vehicle's method
            # Note: vehicle.move also accumulates vehicle.emission, which might be redundant.
            # We use update_emission here as it likely represents the intended per-step value for reward.
            if not vehicle.crashed:
                emissions_this_step += vehicle.update_emission()
                # Calculate sum of squared waiting times for waiting vehicles
                if vehicle.waiting_time > 0:
                    sum_sq_waiting_time += vehicle.waiting_time**2

        # Calculate average speed
        avg_speed = sum_speed / vehicle_count if vehicle_count > 0 else 0
        # --- End Metric Calculation --- #

        # --- Simulation Update (Movement & Cleanup) --- #
        crashes_this_step = 0  # Reset crash count for the step
        vehicles_to_remove = []
        newly_crossed_count = 0  # Track vehicles crossing *this* step

        # --- Spatial Grid Update (Conceptual) --- #
        # spatial_grid = update_spatial_grid(simulation) # Update grid before move checks
        # spatial_grid = None  # Placeholder // REMOVED, grid is now passed
        # --- End Spatial Grid Update --- #

        for vehicle in list(simulation):  # Iterate over a copy for safe removal
            crashes_result = 0
            if not vehicle.crashed:
                # Pass the *populated* spatial grid to the move function # REMOVED spatial_grid argument
                crashes_result = vehicle.move(
                    vehicles,
                    active_lights,
                    stopLines,
                    movingGap,
                    simulation,
                    # spatial_grid, # REMOVED Argument
                )
                crashes_this_step += crashes_result
                # Check if vehicle crossed the line *during this move*
                if vehicle.crossed == 1 and not hasattr(
                    vehicle, "_crossed_last_step"
                ):  # Track first time crossing
                    newly_crossed_count += 1
                    vehicle._crossed_last_step = True  # Mark it so we don't count again

            # Off-screen check - Use configured dimensions
            off_screen = (
                (vehicle.direction == "right" and vehicle.x > SCREEN_WIDTH)
                or (vehicle.direction == "left" and vehicle.x + vehicle.width < 0)
                or (vehicle.direction == "down" and vehicle.y > SCREEN_HEIGHT)
                or (vehicle.direction == "up" and vehicle.y + vehicle.height < 0)
            )

            # Mark for removal if crashed OR off_screen
            if vehicle.crashed or off_screen:
                vehicles_to_remove.append(vehicle)

        # Second pass: Remove marked vehicles safely
        for vehicle in vehicles_to_remove:
            simulation.remove(vehicle)
            try:
                # Remove from the main vehicles dictionary
                vehicles[vehicle.direction][vehicle.lane].remove(vehicle)
                # Clean up waiting time entry if it exists
                if vehicle.id in waiting_times[vehicle.direction]:
                    del waiting_times[vehicle.direction][vehicle.id]
            except (ValueError, KeyError) as e:
                # Log warning but continue if removal fails (vehicle might already be gone)
                logger.warning(
                    f"Failed to remove vehicle {getattr(vehicle, 'id', '?')} from structure: {e}"
                )
                pass
        # --- End Simulation Update --- #

        # --- Neural Controller Update (Call the original update method) --- #
        # Get state based on vehicles *after* movement and removal
        # REMOVE the manual state/action/reward/store logic
        # current_state = neural_controller.get_state(simulation)
        # ... (removed action determination block) ...
        # ... (removed reward calculation block) ...
        # ... (removed store_transition block) ...
        # ... (removed last_state update) ...
        # ... (removed learn_ppo call - it happens inside update) ...

        # Call the consolidated update method from the agent
        next_active_lights = neural_controller.update(
            simulation,  # Pass the simulation group
            avg_speed,  # Avg speed from *before* move
            crashes_this_step,  # Crashes that happened *during* move
            sum_sq_waiting_time,  # Use sum of squares instead of count
            emissions_this_step,  # Emissions calculated *before* move
            newly_crossed_count,  # Pass number of vehicles that crossed this step
        )
        # --- End Neural Controller Update --- #

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
        # --- End Training Epoch Check --- #

        # Apply the determined action for the next simulation step
        active_lights = next_active_lights

        total_steps += 1
        pbar.update(1)

        if not PERFORMANCE_MODE:
            logger.info("Waiting for SPACE key press to continue...")
            while neural_controller.waiting_for_space:
                if neural_controller.check_for_space():
                    break  # Exit wait loop once space is pressed
                time.sleep(0.1)  # Prevent busy-waiting

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
