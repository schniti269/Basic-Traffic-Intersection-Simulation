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
    simulation,
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
        pygame.display.set_caption("TRAFFIC RL SIMULATION")
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
            vehicle.move()  # Move vehicles based on simulation logic
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

            # display the vehicles if rendering
            for vehicle in simulation:
                screen.blit(vehicle.image, [vehicle.x, vehicle.y])
                # vehicle.move() # Moved vehicle movement outside rendering block

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
