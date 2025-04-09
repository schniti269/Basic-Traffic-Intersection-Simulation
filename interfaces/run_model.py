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
    NeuralTrafficControllerPPO,
    get_vehicles_in_zones,
    DEFAULT_SCAN_ZONE_CONFIG,
)

# --- Simulation Configuration ---
VISUAL_SIM_STEPS = 20000
DEFAULT_MODEL_PREFIX = (
    None  # Set to a path prefix like "saved_models/model_final_..." if needed
)
# --- End Configuration ---


def run_visual_simulation(actor_path):
    """Runs the traffic simulation with visuals using a trained PPO model prefix."""

    # Create PPO controller
    neural_controller = NeuralTrafficControllerPPO(steps_per_epoch=1, total_epochs=1)
    neural_controller.load_model(actor_path)  # Load using the prefix
    neural_controller.render_interval = 1  # Ensure rendering is always on
    neural_controller.show_render = True

    # Start background threads
    thread1 = threading.Thread(name="initialization", target=initialize, args=())
    thread1.daemon = True
    thread1.start()
    logger.info("Initialization thread started")

    thread2 = threading.Thread(
        name="generateVehicles", target=generateVehicles, args=()
    )
    thread2.daemon = True
    thread2.start()
    logger.info("Vehicle generation thread started")

    # --- Pygame Initialization --- #
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    try:
        background = pygame.image.load("images/intersection.png")
        screen = pygame.display.set_mode(screenSize)
        pygame.display.set_caption(
            "TRAFFIC SIMULATION - PPO MODEL RUN"
        )  # Updated caption
        redSignal = pygame.image.load("images/signals/red.png")
        yellowSignal = pygame.image.load("images/signals/yellow.png")
        greenSignal = pygame.image.load("images/signals/green.png")
        font = pygame.font.Font(None, 30)
        logger.info("Pygame display initialized for PPO model run.")
    except Exception as e:
        logger.error(f"Error initializing Pygame rendering resources: {e}")
        pygame.quit()
        sys.exit(1)
    # --- End Pygame Init ---

    total_steps = 0
    running = True
    active_lights = [False] * noOfSignals

    while running and total_steps < VISUAL_SIM_STEPS:
        # --- Event Handling --- #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Simulation terminated by user")
                running = False
            elif event.type == pygame.KEYDOWN:
                pass  # Add any specific keybinds here if needed
        # --- End Event Handling --- #

        # --- Calculate Step Metrics (Minimal for Display) --- #
        waiting_vehicles_this_step = 0
        for d in directionNumbers.values():
            for lane in range(3):
                for vehicle in vehicles[d][lane]:
                    if vehicle.waiting_time > 0:
                        waiting_vehicles_this_step += 1
        # --- End Metric Calculation --- #

        # --- Get Action from Loaded PPO Model --- #
        scan_zones = get_vehicles_in_zones(
            directionNumbers, vehicles, DEFAULT_SCAN_ZONE_CONFIG
        )
        current_state = neural_controller.get_state(scan_zones)
        # Use get_action_and_value, but only need the action for simulation
        action_floats, _, _ = neural_controller.get_action_and_value(current_state)
        # Convert action floats (0.0/1.0) to booleans for simulation use
        active_lights = (action_floats > 0.5).tolist()
        # --- End Get Action --- #

        # --- Simulation Update (Movement & Cleanup) --- #
        crashes_this_step = 0
        vehicles_to_remove = []
        for vehicle in list(simulation):
            crashes_result = 0
            if not vehicle.crashed:
                # Using None for spatial_grid as it's not implemented here
                crashes_result = vehicle.move(
                    vehicles, active_lights, stopLines, movingGap, simulation, None
                )
                crashes_this_step += crashes_result

            off_screen = (
                (vehicle.direction == "right" and vehicle.x > screenWidth)
                or (vehicle.direction == "left" and vehicle.x + vehicle.width < 0)
                or (vehicle.direction == "down" and vehicle.y > screenHeight)
                or (vehicle.direction == "up" and vehicle.y + vehicle.height < 0)
            )
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
                    f"Failed to remove vehicle {getattr(vehicle, 'id', '?')} from structure: {e}"
                )
                pass
        # --- End Simulation Update --- #

        # --- Rendering Section --- #
        screen.blit(background, (0, 0))

        info_text = [
            f"Model Prefix: {os.path.basename(model_path)}",  # Updated label
            f"Step: {total_steps}/{VISUAL_SIM_STEPS}",
            f"Waiting Now: {waiting_vehicles_this_step}",
            f"Crashes This Step: {crashes_this_step}",
        ]
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (255, 255, 255), (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 30))

        for i in range(0, noOfSignals):
            sig_img = redSignal if not active_lights[i] else greenSignal
            if sig_img:
                screen.blit(sig_img, signalCoods[i])
            signal_text = "ON" if active_lights[i] else "OFF"
            sig_text_surf = font.render(signal_text, True, (255, 255, 255), (0, 0, 0))
            screen.blit(sig_text_surf, signalTimerCoods[i])

        for vehicle in simulation:
            vehicle.render(screen)

        pygame.display.update()
        clock.tick(60)
        # --- End Rendering Section --- #

        total_steps += 1

    # --- End of Simulation Loop --- #
    logger.info("Visual simulation finished.")
    pygame.quit()


# --- Main Execution --- #
if __name__ == "__main__":
    model_path = r"C:\Users\ian-s\Basic-Traffic-Intersection-Simulation\saved_models\model_epoch_238_reward_-71736_crashes_0_actor.weights.h5"
    run_visual_simulation(model_path)
