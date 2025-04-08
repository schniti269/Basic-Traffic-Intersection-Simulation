import pygame
import sys
import threading
from utils import (
    logger,
    ENABLE_RENDERING,
    SHOW_FINAL_EPOCH_ONLY,
    MAX_EPISODES,
    MANUAL_CONTROL,
    directionNumbers,
    waiting_times,
    crashes,
    signals,
    noOfSignals,
    currentGreen,
    currentYellow,
    signalCoods,
    signalTimerCoods,
    emission_counts,
    simulation,
    vehicles,
    stopLines,
    movingGap,
    defaultStop,
)
from traffic_signal import initialize
from vehicle import generateVehicles
from rl_environment import TrafficEnvironment


# Main simulation class for visualization
class Main:
    # Create RL environment
    env = TrafficEnvironment()

    thread1 = threading.Thread(name="initialization", target=initialize, args=())
    thread1.daemon = True
    thread1.start()
    logger.info("Initialization thread started")

    # Colours
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (255, 0, 0)
    yellow = (255, 255, 0)
    blue = (0, 0, 255)
    purple = (128, 0, 128)
    orange = (255, 165, 0)

    # Screensize
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Initialize Pygame and load assets only if rendering is enabled
    if ENABLE_RENDERING and (
        not SHOW_FINAL_EPOCH_ONLY or env.episodes_completed >= MAX_EPISODES - 1
    ):
        pygame.init()  # Ensure pygame is initialized if rendering
        # Setting background image i.e. image of intersection
        background = pygame.image.load("images/intersection.png")

        screen = pygame.display.set_mode(screenSize)
        pygame.display.set_caption("TRAFFIC SIMULATION")
        logger.info("Pygame display initialized")

        # Loading signal images and font
        redSignal = pygame.image.load("images/signals/red.png")
        yellowSignal = pygame.image.load("images/signals/yellow.png")
        greenSignal = pygame.image.load("images/signals/green.png")
        font = pygame.font.Font(None, 30)
    else:
        # Still need font for potential logging/non-visual output if needed later
        # pygame.font.init() # Required if using fonts without full pygame init
        # font = pygame.font.Font(None, 30) # Keep font if needed elsewhere
        pass  # No screen or images needed if not rendering

    thread2 = threading.Thread(
        name="generateVehicles", target=generateVehicles, args=()
    )
    thread2.daemon = True
    thread2.start()
    logger.info("Vehicle generation thread started")

    simulation_step = 0
    clock = pygame.time.Clock()

    # For manual testing of the RL interface
    if not MANUAL_CONTROL:
        # Reset environment
        state = env.reset()
        action = env.get_action(state)  # Get action from Q-learning agent
        logger.info("RL agent initialized")

    # Main simulation loop
    running = True  # Use a flag to control the loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Simulation terminated by user")
                running = False  # Set flag to false to exit loop
                # sys.exit() # Avoid sys.exit() for cleaner shutdown if possible

        # --- RL Step ---
        if not MANUAL_CONTROL:
            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Update Q-table
            env.update_q_table(state, action, reward, next_state, done)

            # Get next action
            action = env.get_action(next_state)

            # logger.debug(f"Action: {action}, Reward: {reward}, Done: {done}") # Keep logging if desired
            state = next_state

            # Reset if episode is done
            if done:
                logger.info(
                    f"Episode {env.episodes_completed} finished after {env.steps} steps. Total Reward: {env.total_reward}"
                )

                # Check if we've reached the maximum number of episodes
                if env.episodes_completed >= MAX_EPISODES:
                    logger.info(f"Training completed after {MAX_EPISODES} episodes")
                    # Enable rendering for the final epoch if specified
                    if SHOW_FINAL_EPOCH_ONLY and not ENABLE_RENDERING:
                        logger.info("Showing final epoch results")
                        ENABLE_RENDERING = True
                        # Initialize Pygame for final visualization
                        pygame.init()
                        background = pygame.image.load("images/intersection.png")
                        screen = pygame.display.set_mode(screenSize)
                        pygame.display.set_caption(
                            "TRAFFIC RL SIMULATION - FINAL EPOCH"
                        )
                        redSignal = pygame.image.load("images/signals/red.png")
                        yellowSignal = pygame.image.load("images/signals/yellow.png")
                        greenSignal = pygame.image.load("images/signals/green.png")
                        font = pygame.font.Font(None, 30)
                        signalTexts = ["", "", "", ""]  # Initialize signalTexts list
                    elif env.episodes_completed > MAX_EPISODES:
                        # We're done after showing the final epoch
                        running = False
                        logger.info("Final epoch completed. Exiting.")
                        break

                state = env.reset()
                action = env.get_action(state)
        # --- End RL Step ---

        # --- Simulation Update (Movement) ---
        # This needs to run regardless of rendering
        for vehicle in simulation:
            vehicle.move(
                vehicles, currentGreen, currentYellow, stopLines, movingGap
            )  # Move vehicles based on simulation logic
        # --- End Simulation Update ---

        # --- Rendering Section ---
        if ENABLE_RENDERING and (
            not SHOW_FINAL_EPOCH_ONLY or env.episodes_completed >= MAX_EPISODES
        ):
            screen.blit(background, (0, 0))  # display background in simulation

            # Always initialize signalTexts at the beginning of rendering section
            signalTexts = ["", "", "", ""]

            # Display RL metrics if rendering
            if not MANUAL_CONTROL:
                metrics_text = [
                    f"Episode: {env.episodes_completed}/{MAX_EPISODES}",
                    f"Total Reward: {env.total_reward:.2f}",
                    f"Crashes: {crashes}",
                    f"Step: {env.steps}",
                    f"Waiting Cars: {sum([len(waiting_times[d]) for d in directionNumbers.values()])}",
                    f"Q-Table Size: {len(env.q_table)}",
                ]

                # Add emissions by vehicle type
                for vehicle_class in ["car", "bus", "truck", "bike"]:
                    emission_sum = 0
                    for direction in directionNumbers.values():
                        emission_sum += emission_counts[direction][vehicle_class]
                    metrics_text.append(
                        f"{vehicle_class.capitalize()} Emissions: {emission_sum:.1f}"
                    )

                for i, text in enumerate(metrics_text):
                    text_surface = font.render(text, True, white, black)
                    screen.blit(text_surface, (10, 10 + i * 30))
            else:
                # Display manual control metrics
                metrics_text = [
                    f"Manual Control Mode",
                    f"Crashes: {crashes}",
                    f"Waiting Cars: {sum([len(waiting_times[d]) for d in directionNumbers.values()])}",
                ]

                for i, text in enumerate(metrics_text):
                    text_surface = font.render(text, True, white, black)
                    screen.blit(text_surface, (10, 10 + i * 30))

            # Display signals and timers if rendering
            for i in range(0, noOfSignals):
                if i == currentGreen:
                    if currentYellow == 1:
                        signals[i].signalText = signals[i].yellow
                        screen.blit(yellowSignal, signalCoods[i])
                    else:
                        signals[i].signalText = signals[i].green
                        screen.blit(greenSignal, signalCoods[i])
                else:
                    if signals[i].red <= 10:
                        signals[i].signalText = signals[i].red
                    else:
                        signals[i].signalText = "---"
                    screen.blit(redSignal, signalCoods[i])

                signalTexts[i] = font.render(
                    str(signals[i].signalText), True, white, black
                )
                screen.blit(signalTexts[i], signalTimerCoods[i])

            # Draw stop lines with visual indicators
            # Right direction stop line
            pygame.draw.line(
                screen, red, (stopLines["right"], 300), (stopLines["right"], 450), 3
            )
            pygame.draw.rect(screen, red, (stopLines["right"] - 5, 300, 10, 150), 2)

            # Down direction stop line
            pygame.draw.line(
                screen, red, (700, stopLines["down"]), (900, stopLines["down"]), 3
            )
            pygame.draw.rect(screen, red, (700, stopLines["down"] - 5, 200, 10), 2)

            # Left direction stop line
            pygame.draw.line(
                screen, red, (stopLines["left"], 400), (stopLines["left"], 550), 3
            )
            pygame.draw.rect(screen, red, (stopLines["left"] - 5, 400, 10, 150), 2)

            # Up direction stop line
            pygame.draw.line(
                screen, red, (500, stopLines["up"]), (700, stopLines["up"]), 3
            )
            pygame.draw.rect(screen, red, (500, stopLines["up"] - 5, 200, 10), 2)

            # Draw default stop positions
            # Right direction default stop
            pygame.draw.line(
                screen,
                yellow,
                (defaultStop["right"], 300),
                (defaultStop["right"], 450),
                2,
            )

            # Down direction default stop
            pygame.draw.line(
                screen,
                yellow,
                (700, defaultStop["down"]),
                (900, defaultStop["down"]),
                2,
            )

            # Left direction default stop
            pygame.draw.line(
                screen,
                yellow,
                (defaultStop["left"], 400),
                (defaultStop["left"], 550),
                2,
            )

            # Up direction default stop
            pygame.draw.line(
                screen, yellow, (500, defaultStop["up"]), (700, defaultStop["up"]), 2
            )

            # Add labels for stop lines and default stops
            stop_line_font = pygame.font.Font(None, 24)

            # Right direction labels
            stop_line_text = stop_line_font.render("Stop Line", True, white)
            screen.blit(stop_line_text, (stopLines["right"] - 40, 280))
            default_stop_text = stop_line_font.render("Default Stop", True, white)
            screen.blit(default_stop_text, (defaultStop["right"] - 60, 280))

            # Down direction labels
            stop_line_text = stop_line_font.render("Stop Line", True, white)
            screen.blit(stop_line_text, (720, stopLines["down"] - 20))
            default_stop_text = stop_line_font.render("Default Stop", True, white)
            screen.blit(default_stop_text, (720, defaultStop["down"] - 20))

            # Left direction labels
            stop_line_text = stop_line_font.render("Stop Line", True, white)
            screen.blit(stop_line_text, (stopLines["left"] - 40, 380))
            default_stop_text = stop_line_font.render("Default Stop", True, white)
            screen.blit(default_stop_text, (defaultStop["left"] - 60, 380))

            # Up direction labels
            stop_line_text = stop_line_font.render("Stop Line", True, white)
            screen.blit(stop_line_text, (520, stopLines["up"] - 20))
            default_stop_text = stop_line_font.render("Default Stop", True, white)
            screen.blit(default_stop_text, (520, defaultStop["up"] - 20))

            # display the vehicles if rendering
            for vehicle in simulation:
                vehicle.render(
                    screen
                )  # Use the vehicle's render method which includes waiting time display

            pygame.display.update()
        # --- End Rendering Section ---

        # Control simulation speed - increase speed during training
        if not ENABLE_RENDERING or env.episodes_completed < MAX_EPISODES:
            clock.tick(600)  # Run much faster during training
        else:
            clock.tick(60)  # Normal speed for visualization

    # Clean up Pygame if it was initialized
    if ENABLE_RENDERING:
        pygame.quit()
    sys.exit()  # Exit after loop finishes


# For direct execution as a script
if __name__ == "__main__":
    Main()
