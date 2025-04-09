import pygame
import sys
import threading
import argparse
import os

# Updated imports:
import sys

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
    signalCoods,  # Needed for rendering
    signalTimerCoods,  # Needed for rendering
)
from core.simulation.traffic_signal import initialize
from core.simulation.vehicle import generateVehicles

# Removed: from rl_environment import TrafficEnvironment, DEFAULT_SCAN_ZONE_CONFIG
from core.agent.neural_model_01 import (
    NeuralTrafficController,
    get_vehicles_in_zones,
    DEFAULT_SCAN_ZONE_CONFIG,
)  # Moved DEFAULT_SCAN_ZONE_CONFIG here

# --- Simulation Configuration ---
VISUAL_SIM_STEPS = 20000  # How many steps to run the visual simulation for
DEFAULT_MODEL_PATH = (
    None  # Set to a path like "saved_models/model_final_... .weights.h5" if needed
)
# --- End Configuration ---


def run_visual_simulation(model_path):
    """Runs the traffic simulation with visuals using a trained model."""
    logger.info(f"Starting visual simulation using model: {model_path}")

    if not os.path.exists(model_path):
        # Check if path relative to saved_models works
        alt_path = os.path.join("..", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            logger.error(f"Model file not found: {model_path} or {alt_path}")
            sys.exit(1)

    # Initialize environment (conceptually - no separate env class needed now)
    # env = TrafficEnvironment() # Removed

    # Create neural network controller (architecture must match saved weights)
    # Use dummy steps/epochs as training parameters are not relevant here
    neural_controller = NeuralTrafficController(steps_per_epoch=1, total_epochs=1)
    neural_controller.load_model(model_path)  # Load the trained weights
    neural_controller.render_interval = (
        1  # Ensure rendering is always on for this script
    )
    neural_controller.show_render = True

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

    # --- Pygame Initialization for Visuals ---
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    try:
        background = pygame.image.load("images/intersection.png")
        screen = pygame.display.set_mode(screenSize)
        pygame.display.set_caption("TRAFFIC SIMULATION - MODEL RUN")
        redSignal = pygame.image.load("images/signals/red.png")
        yellowSignal = pygame.image.load("images/signals/yellow.png")
        greenSignal = pygame.image.load("images/signals/green.png")
        font = pygame.font.Font(None, 30)
        logger.info("Pygame display initialized for model run.")
    except Exception as e:
        logger.error(f"Error initializing Pygame rendering resources: {e}")
        pygame.quit()
        sys.exit(1)
    # --- End Pygame Init ---

    total_steps = 0
    running = True
    active_lights = [False] * noOfSignals  # Initial light state

    while running and total_steps < VISUAL_SIM_STEPS:
        # --- Event Handling --- #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Simulation terminated by user")
                running = False
            elif event.type == pygame.KEYDOWN:
                # Add any run-specific keybinds if needed (e.g., pause, speed change)

                # References to non-existent 'env' commented out:
                # if event.key == pygame.K_c:
                #     env.show_coordinates = not env.show_coordinates
                #     logger.info(
                #         f"Coordinate display {'enabled' if env.show_coordinates else 'disabled'}"
                #     )

                # Keybinds 1-4 are only for debugging training

                pass  # Added pass to satisfy linter after commenting above block

        # --- End Event Handling --- #

        # --- Calculate Step Metrics (Minimal - only needed if displayed) --- #
        waiting_vehicles_this_step = 0
        for d in directionNumbers.values():
            for lane in range(3):
                for vehicle in vehicles[d][lane]:
                    if vehicle.waiting_time > 0:
                        waiting_vehicles_this_step += 1
        # --- End Metric Calculation --- #

        # --- Get Action from Loaded Model --- #
        scan_zones = get_vehicles_in_zones(
            directionNumbers, vehicles, DEFAULT_SCAN_ZONE_CONFIG
        )
        # Convert scan_zones to state representation first
        current_state = neural_controller.get_state(scan_zones)
        # Use the loaded model to get the action from the state
        # We don't need the probabilities here, just the thresholded action
        active_lights, _ = neural_controller.get_action(current_state)
        # --- End Get Action --- #

        # --- Simulation Update (Movement & Cleanup) --- #
        crashes_this_step = 0  # Reset counter for display this step
        vehicles_to_remove = []
        for vehicle in list(simulation):
            # Remove the initial skip check for crashed vehicles
            # if vehicle.crashed:
            #     continue
            old_x, old_y = vehicle.x, vehicle.y
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

        # --- Rendering Section --- #
        screen.blit(background, (0, 0))

        # Display run info
        info_text = [
            f"Model: {os.path.basename(model_path)}",
            f"Step: {total_steps}/{VISUAL_SIM_STEPS}",
            f"Waiting Now: {waiting_vehicles_this_step}",
            f"Crashes This Step: {crashes_this_step}",
        ]
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (255, 255, 255), (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 30))

        # Display signals
        signalTexts = ["", "", "", ""]
        for i in range(0, noOfSignals):
            sig_img = redSignal if not active_lights[i] else greenSignal
            if sig_img:
                screen.blit(sig_img, signalCoods[i])
            signal_text = "ON" if active_lights[i] else "OFF"
            sig_text_surf = font.render(signal_text, True, (255, 255, 255), (0, 0, 0))
            screen.blit(sig_text_surf, signalTimerCoods[i])

        # Render vehicles
        for vehicle in simulation:
            vehicle.render(screen)

        # Visualize scan zones if needed
        # Commented out reference to non-existent env
        # if env.show_coordinates:
        #     env.visualize_scan_zone(screen)

        pygame.display.update()
        clock.tick(60)  # Run at normal visual speed
        # --- End Rendering Section --- #

        total_steps += 1

    # --- End of Simulation Loop --- #
    logger.info("Visual simulation finished.")
    pygame.quit()


# --- Main Execution --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Traffic Simulation with a trained model."
    )
    parser.add_argument(
        "model_path",
        nargs="?",  # Make argument optional
        default=DEFAULT_MODEL_PATH,  # Use default if not provided
        help="Path to the saved model weights file (.weights.h5)",
    )
    args = parser.parse_args()

    if args.model_path is None:
        # Try to find the latest model in the default directory if no path is given
        logger.warning(
            "No model path provided. Attempting to find latest model in saved_models/..."
        )
        try:
            models_dir = "saved_models"
            all_models = [
                os.path.join(models_dir, f)
                for f in os.listdir(models_dir)
                if f.endswith(".weights.h5")
            ]
            if not all_models:
                logger.error(
                    f"No models found in {models_dir}. Please provide a model path or train a model first."
                )
                sys.exit(1)
            latest_model = max(all_models, key=os.path.getctime)
            logger.info(f"Using latest model found: {latest_model}")
            args.model_path = latest_model
        except FileNotFoundError:
            logger.error(
                f"Directory 'saved_models' not found. Please provide a model path or train a model first."
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error finding latest model: {e}")
            sys.exit(1)

    try:
        run_visual_simulation(args.model_path)
    except Exception as e:
        logger.exception(f"An error occurred during simulation run: {e}")
    finally:
        pygame.quit()  # Ensure Pygame quits cleanly
        sys.exit()  # Ensure script exits

if __name__ == "__main__":
    modelpath = r"saved_models\model_final_epoch_5_reward_0_crashes_0.weights.h5"
    run_visual_simulation(modelpath)
