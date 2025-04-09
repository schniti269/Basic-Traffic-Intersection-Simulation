import pygame
import sys
import threading
import os
from tqdm import tqdm
import sys
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.utils import (
    logger,
    # ENABLE_RENDERING, # No longer needed directly, controller handles rendering flag
    # SHOW_FINAL_EPOCH_ONLY, # No longer needed directly
    # MAX_EPISODES, # Not used
    # MANUAL_CONTROL, # Not used
    directionNumbers,
    waiting_times,
    # crashes, # Calculated per step
    # signals, # Handled by controller/rendering
    noOfSignals,
    # currentGreen, # Handled by controller
    # currentYellow, # Handled by controller
    # signalCoods, # Rendering specific
    # signalTimerCoods, # Rendering specific
    # emission_counts, # Calculated per step
    simulation,
    vehicles,
    stopLines,
    movingGap,
    defaultStop,
    PERFORMANCE_MODE,
)
from core.simulation.traffic_signal import initialize
from core.simulation.vehicle import generateVehicles
from core.agent.neural_model_01 import (
    NeuralTrafficController,
    get_vehicles_in_zones,
    DEFAULT_SCAN_ZONE_CONFIG,
)

# --- Training Configuration ---
MODEL_SAVE_DIR = "saved_models"
STEPS_PER_EPOCH = 10000
TOTAL_EPOCHS = 5
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

    # --- Pygame Initialization (Minimal for Headless) ---
    # We need pygame initialized for Clock and event checking, even if no display
    # Avoid initializing full display unless rendering is triggered
    pygame.init()
    pygame.font.init()  # Needed for potential debug text even if no screen
    pygame.display.init()  # Initialize the display system unconditionally
    clock = pygame.time.Clock()
    screenWidth = 1400  # Still needed for off-screen checks
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)
    # Variables for potential rendering if triggered
    screen = None
    background = None
    redSignal = None
    yellowSignal = None
    greenSignal = None
    font = None
    # --- End Pygame Init ---

    total_steps = 0
    running = True
    active_lights = [False] * noOfSignals  # Initial light state
    should_render = neural_controller.should_render()  # Initial render check

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

        # --- Minimal Event Handling (Needed for QUIT and SPACE) --- #
        # Poll events always, but only process space if waiting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Training terminated by user")
                running = False
                break  # Exit event loop
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and neural_controller.waiting_for_space:
                    if neural_controller.check_for_space():
                        should_render = neural_controller.should_render()
        if not running:
            break  # Exit main loop
        # --- End Event Handling --- #

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
            old_x, old_y = vehicle.x, vehicle.y  # Store pos before move
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

            # Off-screen check
            off_screen = (
                (vehicle.direction == "right" and vehicle.x > screenWidth)
                or (vehicle.direction == "left" and vehicle.x + vehicle.width < 0)
                or (vehicle.direction == "down" and vehicle.y > screenHeight)
                or (vehicle.direction == "up" and vehicle.y + vehicle.height < 0)
            )
            # Check for removal *after* potential move/crash flag update
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

            # Update rendering flag and manage pygame display
            should_render = neural_controller.should_render()
            if should_render and screen is None:
                logger.info("Initializing Pygame display for rendering...")
                # Load resources needed for rendering
                try:
                    background = pygame.image.load(
                        "images/intersection.png"
                    )  # Corrected path
                    screen = pygame.display.set_mode(screenSize)
                    pygame.display.set_caption("TRAFFIC SIMULATION - TRAINING VIS")
                    redSignal = pygame.image.load("../images/signals/red.png")
                    yellowSignal = pygame.image.load("../images/signals/yellow.png")
                    greenSignal = pygame.image.load("../images/signals/green.png")
                    font = pygame.font.Font(None, 30)
                    # Force vehicles to load images
                    for v in simulation:
                        v.load_image_if_needed()
                except Exception as e:
                    logger.error(f"Error initializing Pygame rendering resources: {e}")
                    should_render = False  # Fallback to headless if resources fail
                    pygame.display.quit()
                    pygame.font.quit()
                    pygame.display.init()  # Re-init display for headless operation
                    pygame.font.init()
            elif not should_render and screen is not None:  # Check screen is not None
                logger.info("Quitting Pygame display for headless mode...")
                pygame.display.quit()
                pygame.display.init()  # Keep display initialized for events/image loading
                screen = None  # Reset screen variable
        # --- End Training Epoch Check --- #

        # --- Rendering Section --- #
        if should_render and pygame.display.get_init() and screen is not None:
            try:
                screen.blit(background, (0, 0))
                # ... (rest of rendering logic similar to run_model.py) ...
                # Display neural metrics
                metrics_text = [
                    f"Epoch: {neural_controller.current_epoch}",
                    f"Steps: {total_steps} (Epoch Step: {neural_controller.epoch_steps})",
                    f"Current Epoch Reward: {neural_controller.epoch_accumulated_reward:.2f}",
                    f"Crashes (Epoch): {neural_controller.total_crashes_epoch}",
                    f"Waiting Now: {waiting_vehicles_this_step}",
                    f"Emissions Now: {emissions_this_step:.2f}",
                ]
                # Add more metrics if needed
                if font:
                    for i, text in enumerate(metrics_text):
                        text_surface = font.render(
                            text, True, (255, 255, 255), (0, 0, 0)
                        )
                        screen.blit(text_surface, (10, 10 + i * 30))
                else:
                    logger.warning("Font not loaded for rendering metrics.")

                # Display signals
                for i in range(0, noOfSignals):
                    sig_img = redSignal if not active_lights[i] else greenSignal
                    if sig_img:
                        screen.blit(sig_img, signalCoods[i])
                    # Add text ON/OFF if font loaded
                    if font:
                        signal_text = "ON" if active_lights[i] else "OFF"
                        sig_text_surf = font.render(
                            signal_text, True, (255, 255, 255), (0, 0, 0)
                        )
                        screen.blit(sig_text_surf, signalTimerCoods[i])

                # Render vehicles
                for vehicle in simulation:
                    vehicle.render(screen)

                # Visualize scan zones if env available
                if "env" in locals() and hasattr(env, "visualize_scan_zone"):
                    env.visualize_scan_zone(screen)

                # Space prompt
                if neural_controller.waiting_for_space and font:
                    space_text = "Press SPACE to continue training"
                    space_surface = font.render(
                        space_text, True, (255, 255, 255), (0, 0, 0)
                    )
                    screen.blit(space_surface, (500, 40))

                pygame.display.update()
                clock.tick(60)  # Limit speed when rendering
            except Exception as e:
                logger.error(f"Error during rendering: {e}")
                # Optionally disable rendering for future steps if errors persist
                # should_render = False
                # if pygame.display.get_init(): pygame.display.quit()
        # --- End Rendering Section --- #

        # Use the action determined in the *previous* controller update for the *current* simulation step
        active_lights = next_active_lights

        # Update total steps and progress bar
        total_steps += 1
        if pbar:
            pbar.update(1)

    # --- End of Training Loop --- #
    if pbar:
        pbar.close()

    pygame.quit()  # Quit pygame fully at the end
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
        pygame.quit()  # Ensure Pygame quits cleanly
        sys.exit()  # Ensure script exits
