import pygame
import sys
import argparse
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.utils import (
    logger,
    noOfSignals,
    simulation,
    signalCoods,
    signalTimerCoods,
    crashes,
)
from core.agent.neural_model_01 import (
    NeuralTrafficControllerPPO,
)
from shared.simulation_core import (
    reset_simulation,
    start_simulation_threads, 
    calculate_metrics, 
    update_simulation, 
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)

# --- Simulation Configuration ---
VISUAL_SIM_STEPS = 20000
BENCHMARK_MODE = False
BENCHMARK_STEPS = 1000

def run_visual_simulation(model_prefix, benchmark_mode=BENCHMARK_MODE):
    """Runs the traffic simulation with visuals using a trained PPO model prefix."""

    # Create PPO controller
    neural_controller = NeuralTrafficControllerPPO(steps_per_epoch=1, total_epochs=1)
    neural_controller.load_models(model_prefix)  # Load using the prefix
    if not benchmark_mode:
        neural_controller.render_interval = 1  # Ensure rendering is always on
        neural_controller.show_render = True

    # Reset Simulation zu Beginn
    reset_simulation()

    # Start Simulationsthreads und initialisiere Ampeln auf rot
    active_lights = start_simulation_threads()

    # --- Pygame Initialization --- #
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()
    screenSize = (SCREEN_WIDTH, SCREEN_HEIGHT)

    if not benchmark_mode:
        try:
            background = pygame.image.load("images/intersection.png")
            screen = pygame.display.set_mode(screenSize)
            pygame.display.set_caption(
                "TRAFFIC SIMULATION - PPO MODEL RUN"
            )
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

    # --- Benchmarking Setup ---
    timings = {}
    if benchmark_mode:
        timings = {
            "event_handling": 0.0,
            "metric_calculation": 0.0,
            "simulation_update": 0.0,
            "model_inference": 0.0,
            "rendering": 0.0,
            "total_step_time": 0.0,
        }
        logger.info(f"Starting benchmark mode for {BENCHMARK_STEPS} steps...")

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

        # --- Metric Calculation using central function --- #
        metric_start_time = time.perf_counter() if benchmark_mode else 0
        metrics = calculate_metrics()
        metric_end_time = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["metric_calculation"] += metric_end_time - metric_start_time
        
        # --- Simulation Update using central function --- #
        sim_update_start = time.perf_counter() if benchmark_mode else 0
        update_results = update_simulation(active_lights)
        crashes_this_step = update_results["crashes_this_step"]
        newly_crossed_count = update_results["newly_crossed_count"]
        sim_update_end = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["simulation_update"] += sim_update_end - sim_update_start

        # --- Neural Controller Update --- #
        model_start_time = time.perf_counter() if benchmark_mode else 0
        next_active_lights = neural_controller.update(
            simulation,
            metrics["avg_speed"],
            crashes_this_step,
            metrics["sum_sq_waiting_time"],
            metrics["emissions"],
            newly_crossed_count,
        )
        model_end_time = time.perf_counter() if benchmark_mode else 0
        if benchmark_mode:
            timings["model_inference"] += model_end_time - model_start_time
            
        # Update der Lichter für den nächsten Simulationsschritt
        active_lights = next_active_lights

        # --- Rendering Section --- #
        render_start_time = time.perf_counter() if benchmark_mode else 0
        if not benchmark_mode:
            screen.blit(background, (0, 0))

            info_text = [
                f"Model Prefix: {os.path.basename(model_prefix)}",
                f"Step: {total_steps}/{VISUAL_SIM_STEPS}",
                f"Waiting: {metrics['waiting_vehicles']}",
                f"Crashes This Step: {crashes_this_step}",
                f"Avg Speed: {metrics['avg_speed']:.2f}",
                f"Total Crashes: {crashes}"
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
        total_time = timings.get("total_step_time", 0.0)
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
        required=True,
        help="REQUIRED: Path prefix for the PPO model to load (e.g., saved_models/model_epoch_100)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode (no visuals, timed execution)",
    )
    args = parser.parse_args()

    model_path = args.model
    run_visual_simulation(model_path, benchmark_mode=args.benchmark)
