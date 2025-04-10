import pygame
import sys
import threading
import argparse
import os
import time  # Import time for benchmarking

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
)

# --- Simulation Configuration ---
VISUAL_SIM_STEPS = 20000
# DEFAULT_MODEL_PREFIX = (
#     None  # Set to a path prefix like "saved_models/model_final_..." if needed
# )
BENCHMARK_MODE = False  # Add benchmark mode flag
BENCHMARK_STEPS = 1000  # Number of steps for benchmark

# --- End Configuration ---


def run_visual_simulation(model_prefix, benchmark_mode=BENCHMARK_MODE):
    """Runs the traffic simulation with visuals using a trained PPO model prefix."""

    # Create PPO controller
    neural_controller = NeuralTrafficControllerPPO(steps_per_epoch=1, total_epochs=1)
    neural_controller.load_models(model_prefix)  # Load using the prefix
    if not benchmark_mode:
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

    if not benchmark_mode:
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
    last_active_lights = [False] * noOfSignals  # Initialize persistent light state

    # --- Benchmarking Setup ---
    timings = {}  # Initialize empty timings dict here
    if benchmark_mode:
        # Populate only if in benchmark mode
        timings = {
            "event_handling": 0.0,
            "metric_calculation": 0.0,
            "get_state": 0.0,
            "model_inference": 0.0,
            "vehicle_move": 0.0,
            "vehicle_removal": 0.0,
            "rendering": 0.0,
            "total_step_time": 0.0,
        }
        logger.info(f"Starting benchmark mode for {BENCHMARK_STEPS} steps...")
    # --- End Benchmarking Setup ---

    while running and total_steps < (
        BENCHMARK_STEPS if benchmark_mode else VISUAL_SIM_STEPS
    ):
        step_start_time = time.perf_counter() if benchmark_mode else 0

        # --- Event Handling --- #
        event_start_time = time.perf_counter() if benchmark_mode else 0
        if not benchmark_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Simulation terminated by user")
                    running = False
                elif event.type == pygame.KEYDOWN:
                    pass  # Add any specific keybinds here if needed
        event_end_time = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["event_handling"] += event_end_time - event_start_time
        # --- End Event Handling --- #

        # --- Calculate Step Metrics (Minimal for Display) --- #
        metric_start_time = time.perf_counter() if benchmark_mode else 0
        waiting_vehicles_this_step = 0
        if not benchmark_mode:  # Skip in benchmark for speed
            for d in directionNumbers.values():
                for lane in range(3):
                    for vehicle in vehicles[d][lane]:
                        if vehicle.waiting_time > 0:
                            waiting_vehicles_this_step += 1
        metric_end_time = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["metric_calculation"] += metric_end_time - metric_start_time
        # --- End Metric Calculation --- #

        # --- Get Action from Loaded PPO Model (Run only every 10 steps) --- #
        if total_steps % 10 == 0:  # Only run model every 10 steps
            get_state_start_time = time.perf_counter() if benchmark_mode else 0
            current_state = neural_controller.get_state(simulation)
            get_state_end_time = time.perf_counter() if benchmark_mode else 0

            model_inf_start_time = time.perf_counter() if benchmark_mode else 0
            action_tensor, _, _ = neural_controller.get_action_and_value(current_state)
            action_floats = action_tensor.numpy()  # Convert tensor to numpy array
            current_active_lights = (
                action_floats > 0.5
            ).tolist()  # Calculate new lights
            model_inf_end_time = time.perf_counter() if benchmark_mode else 0

            last_active_lights = (
                current_active_lights  # Store the newly calculated lights
            )

            if benchmark_mode:
                # Accumulate time only when the model runs
                timings["get_state"] = timings.get("get_state", 0.0) + (
                    get_state_end_time - get_state_start_time
                )
                timings["model_inference"] = timings.get("model_inference", 0.0) + (
                    model_inf_end_time - model_inf_start_time
                )
        else:
            # Reuse the last calculated lights for intermediate steps
            current_active_lights = last_active_lights

        # Use the determined lights (either new or reused) for simulation update
        active_lights = current_active_lights

        # --- Simulation Update (Movement & Cleanup) --- #
        move_start_time = time.perf_counter() if benchmark_mode else 0
        crashes_this_step = 0
        vehicles_to_remove = []
        vehicles_to_process = list(simulation)

        for vehicle in vehicles_to_process:
            crashes_result = 0
            if not vehicle.crashed:
                crashes_result = vehicle.move(
                    vehicles, active_lights, stopLines, movingGap, simulation
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
        move_end_time = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["vehicle_move"] += move_end_time - move_start_time

        # --- Remove crashed/off-screen vehicles --- #
        removal_start_time = time.perf_counter() if benchmark_mode else 0
        unique_vehicles_to_remove = set(vehicles_to_remove)

        for vehicle in unique_vehicles_to_remove:
            simulation.remove(vehicle)
            try:
                if vehicle in vehicles[vehicle.direction][vehicle.lane]:
                    vehicles[vehicle.direction][vehicle.lane].remove(vehicle)
                if vehicle.id in waiting_times[vehicle.direction]:
                    del waiting_times[vehicle.direction][vehicle.id]
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Failed to remove vehicle {getattr(vehicle, 'id', '?')} from data structures: {e}"
                )
                pass
        removal_end_time = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["vehicle_removal"] += removal_end_time - removal_start_time
        # --- End Simulation Update --- #

        # --- Rendering Section --- #
        render_start_time = time.perf_counter() if benchmark_mode else 0
        if not benchmark_mode:
            screen.blit(background, (0, 0))

            info_text = [
                f"Model Prefix: {os.path.basename(model_prefix)}",
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
                sig_text_surf = font.render(
                    signal_text, True, (255, 255, 255), (0, 0, 0)
                )
                screen.blit(sig_text_surf, signalTimerCoods[i])

            for vehicle in simulation:
                vehicle.render(screen)

            pygame.display.update()
            clock.tick(60)
        render_end_time = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["rendering"] += render_end_time - render_start_time
        # --- End Rendering Section --- #

        step_end_time = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["total_step_time"] += step_end_time - step_start_time

        total_steps += 1

    # --- End of Simulation Loop --- #
    if benchmark_mode:
        logger.info("--- Benchmark Results ---")
        # No need to re-initialize here, use the populated dict
        # timings = {
        #     "event_handling": timings.get("event_handling", 0.0),
        #     "metric_calculation": timings.get("metric_calculation", 0.0),
        #     "get_state": timings.get("get_state", 0.0),
        #     "model_inference": timings.get("model_inference", 0.0),
        #     "vehicle_move": timings.get("vehicle_move", 0.0),
        #     "vehicle_removal": timings.get("vehicle_removal", 0.0),
        #     "rendering": timings.get("rendering", 0.0),
        #     "total_step_time": timings.get("total_step_time", 0.0),
        # }
        total_time = timings.get("total_step_time", 0.0)  # Use .get for safety
        if BENCHMARK_STEPS > 0:
            avg_step_time = total_time / BENCHMARK_STEPS
            logger.info(
                f"Total time for {BENCHMARK_STEPS} steps: {total_time:.4f} seconds"
            )
            logger.info(f"Average time per step: {avg_step_time:.6f} seconds")
            logger.info("Average time per section:")
            for section, time_spent in timings.items():
                if section != "total_step_time":
                    avg_section_time = time_spent / BENCHMARK_STEPS
                    percentage = (
                        (time_spent / total_time * 100) if total_time > 0 else 0
                    )
                    logger.info(
                        f"  - {section}: {avg_section_time:.6f} seconds ({percentage:.2f}%)"
                    )
        else:
            logger.warning("Benchmark ran for 0 steps. Cannot calculate averages.")
        logger.info("-------------------------")
    else:
        logger.info("Visual simulation finished.")
        pygame.quit()


# --- Main Execution --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Traffic Simulation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,  # Make model path required
        help="REQUIRED: Path prefix for the PPO model to load (e.g., saved_models/model_epoch_100)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",  # Use action='store_true'
        help="Run in benchmark mode (no visuals, timed execution)",
    )
    args = parser.parse_args()

    model_path = args.model
    # if model_path is None:
    #     # Default logic if --model is not provided
    #     model_path = r"C:\Users\ian-s\Basic-Traffic-Intersection-Simulation\model_epoch_3310_reward_11200_crashes_0"
    #     logger.info(f"No model specified, using default: {model_path}")

    run_visual_simulation(model_path, benchmark_mode=args.benchmark)
