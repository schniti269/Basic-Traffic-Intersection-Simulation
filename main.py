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
from rl_environment import TrafficEnvironment, DEFAULT_SCAN_ZONE_CONFIG
from neural_model_01 import NeuralTrafficController, get_vehicles_in_zones


# Initialize the environment to access scan zone config
env = TrafficEnvironment()

# Define training parameters
STEPS_PER_EPOCH = 100000  # Number of simulation steps per training epoch
TOTAL_EPOCHS = 10  # Total number of epochs to train

# Create neural network controller instead of Q-learning
neural_controller = NeuralTrafficController(
    steps_per_epoch=STEPS_PER_EPOCH, total_epochs=TOTAL_EPOCHS
)


# Function to print vehicle matrix for a given zone
def print_vehicle_matrix(direction):
    zone_name = direction.upper()

    # Get scan zone configuration directly
    scan_zone = DEFAULT_SCAN_ZONE_CONFIG[direction]
    camera = scan_zone["camera"]
    zone = scan_zone["zone"]

    # Log that the matrix was requested
    logger.info(f"Vehicle matrix for {zone_name} zone requested by user")

    # Find all vehicles in the scan zone
    vehicles_in_zone = []
    vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "bike": 0}

    for d in directionNumbers.values():
        for lane in range(3):
            for vehicle in vehicles[d][lane]:
                # Calculate vehicle dimensions
                vehicle_width = vehicle.image.get_rect().width
                vehicle_height = vehicle.image.get_rect().height

                # Calculate vehicle corners
                vehicle_left = vehicle.x
                vehicle_right = vehicle.x + vehicle_width
                vehicle_top = vehicle.y
                vehicle_bottom = vehicle.y + vehicle_height

                # Check if any part of the vehicle is in this scan zone
                in_zone = not (
                    vehicle_right < zone["x1"]
                    or vehicle_left > zone["x2"]
                    or vehicle_bottom < zone["y1"]
                    or vehicle_top > zone["y2"]
                )

                # Calculate distance to camera (signed distance - can be negative if behind camera)
                if direction == "right":
                    # For right direction, we look left from the camera
                    distance = vehicle_left - camera["x"]
                elif direction == "left":
                    # For left direction, we look right from the camera
                    distance = camera["x"] - vehicle_right
                elif direction == "down":
                    # For down direction, we look up from the camera
                    distance = camera["y"] - vehicle_bottom
                else:  # up
                    # For up direction, we look down from the camera
                    distance = vehicle_top - camera["y"]

                # Include all vehicles in the zone
                if in_zone:
                    vehicles_in_zone.append(
                        {
                            "vehicle": vehicle,
                            "distance": abs(
                                distance
                            ),  # Use abs to show distance magnitude
                            "speed": vehicle.speed,
                            "type": vehicle.vehicleClass,
                            "acceleration": (
                                1
                                if vehicle.accelerated
                                else (-1 if vehicle.decelerated else 0)
                            ),
                            "position": (vehicle.x, vehicle.y),
                        }
                    )
                    vehicle_counts[vehicle.vehicleClass] += 1

    # Sort vehicles by distance
    vehicles_in_zone.sort(key=lambda x: x["distance"])

    # Clear visual space before printing
    print("\n\n")
    print("*" * 70)
    print(f"{'*' * 5} VEHICLE MATRIX FOR ZONE: {zone_name} {'*' * 5}")
    print("*" * 70)

    # Print summary information
    print(f"• Total vehicles in zone: {len(vehicles_in_zone)}")
    print(
        f"• Vehicle types: Cars: {vehicle_counts['car']}, Buses: {vehicle_counts['bus']}, "
        + f"Trucks: {vehicle_counts['truck']}, Bikes: {vehicle_counts['bike']}"
    )
    print(
        f"• Zone coordinates: ({zone['x1']}, {zone['y1']}) to ({zone['x2']}, {zone['y2']})"
    )
    print(f"• Camera position: ({camera['x']}, {camera['y']})")

    # Print vehicle information sorted by distance
    if len(vehicles_in_zone) > 0:
        print("\nVEHICLES BY DISTANCE TO CAMERA:")
        print(f"{'#':<4} {'Type':<10} {'Distance':<10} {'Speed':<10} {'Waiting':<10}")
        print("-" * 55)

        for i, vehicle_data in enumerate(vehicles_in_zone, 1):
            v = vehicle_data["vehicle"]
            print(
                f"{i:<4} {vehicle_data['type']:<10} {vehicle_data['distance']:<10.1f} "
                + f"{vehicle_data['speed']:<10.1f} {v.waiting_time:<10.1f}"
            )
    else:
        print("\nNo vehicles detected in this zone.")

    # Clear visual space after printing
    print("\n" + "*" * 70 + "\n")


# Main simulation class for visualization
class Main:
    # Create neural controller
    neural_controller = neural_controller

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
    should_render = neural_controller.should_render()

    if should_render:
        pygame.init()  # Ensure pygame is initialized if rendering
        # Setting background image i.e. image of intersection
        background = pygame.image.load("images/intersection.png")

        screen = pygame.display.set_mode(screenSize)
        pygame.display.set_caption("TRAFFIC SIMULATION - NEURAL CONTROLLER")
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

    # For neural controller
    total_steps = 0
    epoch_steps = 0
    training_sample_size = 1000  # Steps per training sample
    running = True  # Use a flag to control the loop
    global crashes  # Declare crashes as global before the loop starts

    # Main simulation loop
    while running:
        # DEBUG: Check if main loop is running
        if total_steps % 1000 == 0:  # Print less frequently than step update
            print(f"Main loop running - Step: {total_steps}", flush=True)

        # Get all pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Simulation terminated by user")
                running = False  # Set flag to false to exit loop
            elif event.type == pygame.KEYDOWN:
                # Toggle coordinate display with 'C' key
                if event.key == pygame.K_c:
                    env.show_coordinates = not env.show_coordinates
                    logger.info(
                        f"Coordinate display {'enabled' if env.show_coordinates else 'disabled'}"
                    )
                # Print vehicle matrix for specific zones (1-4 keys)
                elif event.key == pygame.K_1:
                    print_vehicle_matrix("right")
                elif event.key == pygame.K_2:
                    print_vehicle_matrix("down")
                elif event.key == pygame.K_3:
                    print_vehicle_matrix("left")
                elif event.key == pygame.K_4:
                    print_vehicle_matrix("up")
                # Space key to continue after visualization
                elif (
                    event.key == pygame.K_SPACE and neural_controller.waiting_for_space
                ):
                    neural_controller.waiting_for_space = False
                    neural_controller.show_render = False
                    logger.info("Continuing training after visualization")

        # --- Neural Controller Step ---
        # Get data from all 4 scan zones
        scan_zones = get_vehicles_in_zones(
            directionNumbers, vehicles, DEFAULT_SCAN_ZONE_CONFIG
        )

        # Calculate metrics for reward
        total_waiting = sum([len(waiting_times[d]) for d in directionNumbers.values()])
        avg_speed = 0
        vehicle_count = 0

        # Calculate average speed across all vehicles
        for d in directionNumbers.values():
            for lane in range(3):
                for vehicle in vehicles[d][lane]:
                    avg_speed += vehicle.speed
                    vehicle_count += 1

        if vehicle_count > 0:
            avg_speed /= vehicle_count

        # Calculate total emissions
        total_emissions = 0
        for direction in directionNumbers.values():
            for vehicle_class in ["car", "bus", "truck", "bike"]:
                total_emissions += emission_counts[direction][vehicle_class]

        # DEBUG: Check if controller update is reached
        print(f"Step {total_steps}: Calling controller update...", flush=True)
        active_lights = neural_controller.update(
            scan_zones,
            avg_speed,
            crashes,
            total_waiting,
            total_emissions,
            vehicle_count,
        )

        # Update step counts
        total_steps += 1
        epoch_steps += 1

        # Check if we've completed a training epoch
        if epoch_steps >= neural_controller.steps_per_epoch:
            neural_controller.end_epoch()
            epoch_steps = 0

            # Reset waiting times and emission counts
            for direction in directionNumbers.values():
                waiting_times[direction] = {}
                emission_counts[direction] = {
                    "car": 0,
                    "bus": 0,
                    "truck": 0,
                    "bike": 0,
                    "total": 0,
                }

            # Reset global crashes count
            crashes = 0

            # Update rendering flag based on controller state
            should_render = neural_controller.should_render()

            # Initialize rendering if needed
            if should_render and not ENABLE_RENDERING:
                ENABLE_RENDERING = True
                pygame.init()
                background = pygame.image.load("images/intersection.png")
                screen = pygame.display.set_mode(screenSize)
                pygame.display.set_caption("TRAFFIC SIMULATION - NEURAL CONTROLLER")
                redSignal = pygame.image.load("images/signals/red.png")
                yellowSignal = pygame.image.load("images/signals/yellow.png")
                greenSignal = pygame.image.load("images/signals/green.png")
                font = pygame.font.Font(None, 30)
            elif not should_render and ENABLE_RENDERING:
                ENABLE_RENDERING = False

        # Wait for space key if needed
        if neural_controller.waiting_for_space:
            # Just keep checking for space without advancing simulation
            continue
        # --- End Neural Controller Step ---

        # --- Simulation Update (Movement) ---
        # Pass the list of active lights directly to the move function
        # The move function needs to be updated to handle this list
        # We also need to remove the currentYellow logic for now, or adapt it.
        # Let's assume the vehicle move logic can handle the active_lights list.
        for vehicle in simulation:
            # We need to modify vehicle.move to accept active_lights instead of currentGreen/currentYellow
            vehicle.move(vehicles, active_lights, stopLines, movingGap)
        # --- End Simulation Update ---

        # --- Rendering Section ---
        if ENABLE_RENDERING:
            screen.blit(background, (0, 0))  # display background in simulation

            # Always initialize signalTexts at the beginning of rendering section
            signalTexts = ["", "", "", ""]

            # Display neural metrics if rendering
            metrics_text = [
                f"Epoch: {neural_controller.current_epoch}",
                f"Steps: {total_steps}",
                f"Current Reward: {neural_controller.epoch_reward:.2f}",
                f"Crashes: {crashes}",
                f"Waiting Cars: {total_waiting}",
                f"Epsilon: {neural_controller.epsilon:.3f}",
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

            # Add key instruction text
            instruction_text = (
                "Press keys 1-4 to print vehicle matrices for each zone in the console"
            )
            instruction_surface = font.render(instruction_text, True, white, black)
            screen.blit(instruction_surface, (500, 10))

            if neural_controller.waiting_for_space:
                space_text = "Press SPACE to continue training"
                space_surface = font.render(space_text, True, white, black)
                screen.blit(space_surface, (500, 40))

            # Visualize scan zones
            env.visualize_scan_zone(screen)

            # Display signals based on the active_lights list
            # Remove the old timer logic for simplicity with multi-light activation
            for i in range(0, noOfSignals):
                if active_lights[i]:  # Check the boolean list from the controller
                    screen.blit(greenSignal, signalCoods[i])
                    signal_text = "ON"
                else:
                    screen.blit(redSignal, signalCoods[i])
                    signal_text = "OFF"

                # Display simple ON/OFF text instead of timer
                signalTexts[i] = font.render(signal_text, True, white, black)
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

            # Render vehicles
            for vehicle in simulation:
                vehicle.render(screen)

            # Display zone key mapping indicators
            zone_font = pygame.font.Font(None, 24)
            zone_mappings = [
                ("1: RIGHT ZONE", (1200, 350)),
                ("2: DOWN ZONE", (780, 700)),
                ("3: LEFT ZONE", (200, 430)),
                ("4: UP ZONE", (640, 100)),
            ]

            for text, pos in zone_mappings:
                # Create background for better visibility
                text_surface = zone_font.render(text, True, white)
                text_width, text_height = text_surface.get_size()
                bg_rect = pygame.Rect(
                    pos[0] - 5, pos[1] - 5, text_width + 10, text_height + 10
                )
                pygame.draw.rect(screen, black, bg_rect)
                pygame.draw.rect(screen, orange, bg_rect, 2)

                # Draw zone text
                screen.blit(text_surface, pos)

            pygame.display.update()
        # --- End Rendering Section ---

        # Control simulation speed
        if ENABLE_RENDERING:
            clock.tick(60)  # Normal speed for visualization
        else:
            clock.tick(6000000)  # Run much faster during training

    # Clean up Pygame if it was initialized
    if ENABLE_RENDERING:
        pygame.quit()
    sys.exit()  # Exit after loop finishes


# For direct execution as a script
if __name__ == "__main__":
    Main()
