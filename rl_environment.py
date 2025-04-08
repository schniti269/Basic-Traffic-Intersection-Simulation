import numpy as np
from collections import deque
import random
import pygame
import sys
from utils import (
    vehicles,
    waiting_vehicles,
    emission_counts,
    crashes,
    waiting_times,
    currentGreen,
    currentYellow,
    nextGreen,
    directionNumbers,
    simulation,
    logger,
    WAITING_PENALTY,
    MAX_WAITING_TIME,
    EMISSION_FACTOR,
    CRASH_PENALTY,
    REWARD_SCALE,
    EPSILON,
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    stopLines,
)
from traffic_signal import initialize

# Constants for visualization
SCAN_ZONE_COLOR = (0, 255, 0, 128)  # Semi-transparent green
CAMERA_WINDOW_SIZE = (800, 600)
CAMERA_WINDOW_TITLE = "Traffic Camera View"

# Default scan zone and camera configurations
DEFAULT_SCAN_ZONE_CONFIG = {
    "right": {
        "zone": {
            "x1": 811,  # Left edge
            "y1": 427,  # Top edge
            "x2": 1400,  # Right edge
            "y2": 512,  # Bottom edge
        },
        "camera": {
            "x": 787,  # Camera x position
            "y": 464,  # Camera y position
        },
    },
    "left": {
        "zone": {
            "x1": 0,  # Left edge
            "y1": 370,  # Top edge
            "x2": 580,  # Right edge
            "y2": 424,  # Bottom edge
        },
        "camera": {
            "x": 600,  # Camera x position
            "y": 400,  # Camera y position
        },
    },
    "down": {
        "zone": {
            "x1": 600,  # Left edge
            "y1": 546,  # Top edge
            "x2": 681,  # Right edge
            "y2": 800,  # Bottom edge
        },
        "camera": {
            "x": 730,  # Camera x position
            "y": 330,  # Camera y position
        },
    },
    "up": {
        "zone": {
            "x1": 688,  # Left edge
            "y1": 0,  # Top edge
            "x2": 767,  # Right edge
            "y2": 321,  # Bottom edge
        },
        "camera": {
            "x": 650,  # Camera x position
            "y": stopLines["up"] + 10,  # Camera y position
        },
    },
}


# RL Environment class
class TrafficEnvironment:
    def __init__(self, scan_zone_config=None):
        self.steps = 0
        self.total_reward = 0
        self.episodes_completed = 0
        self.action_space = 4  # Number of possible actions (which signal to turn green)
        self.observation_history = deque(maxlen=5)
        self.q_table = {}  # Q-table for Q-learning

        # Use provided scan zone config or default
        self.scan_zone_config = (
            scan_zone_config if scan_zone_config else DEFAULT_SCAN_ZONE_CONFIG
        )

        # Initialize camera visualization window
        pygame.init()
        self.camera_window = pygame.display.set_mode(CAMERA_WINDOW_SIZE)
        pygame.display.set_caption(CAMERA_WINDOW_TITLE)
        self.camera_font = pygame.font.Font(None, 24)

        # Initialize cursor coordinate display
        self.show_coordinates = True
        self.coordinate_font = pygame.font.Font(None, 24)

        logger.info("TrafficEnvironment initialized")

    def reset(self):
        global vehicles, waiting_vehicles, emission_counts, crashes, waiting_times, currentGreen, currentYellow, nextGreen

        # Reset all environment variables
        vehicles = {
            "right": {0: [], 1: [], 2: [], "crossed": 0},
            "down": {0: [], 1: [], 2: [], "crossed": 0},
            "left": {0: [], 1: [], 2: [], "crossed": 0},
            "up": {0: [], 1: [], 2: [], "crossed": 0},
        }
        waiting_vehicles = {"right": 0, "down": 0, "left": 0, "up": 0}
        emission_counts = {
            "right": {"car": 0, "bus": 0, "truck": 0, "bike": 0, "total": 0},
            "down": {"car": 0, "bus": 0, "truck": 0, "bike": 0, "total": 0},
            "left": {"car": 0, "bus": 0, "truck": 0, "bike": 0, "total": 0},
            "up": {"car": 0, "bus": 0, "truck": 0, "bike": 0, "total": 0},
        }
        waiting_times = {"right": {}, "down": {}, "left": {}, "up": {}}
        crashes = 0
        self.total_reward = 0
        self.steps = 0
        currentGreen = 0
        currentYellow = 0
        nextGreen = 1

        self.episodes_completed += 1
        logger.info(f"Episode {self.episodes_completed} started")

        # Clear simulation
        simulation.empty()

        # Initialize signals
        initialize()

        logger.info("Environment reset")

        # Return initial observation
        return self.get_state()

    def get_camera_state(self):
        """
        Returns a vectorized representation of vehicles in the camera's field of view.
        The camera is positioned at the traffic light and looks at approaching vehicles.
        Returns a fixed-size vector of shape (50, 5) where each row represents:
        [distance_to_light, speed, acceleration, vehicle_type, is_empty_flag]
        """
        MAX_VEHICLES = 50
        VEHICLE_TYPE_MAP = {"car": 0, "bus": 1, "truck": 2, "bike": 3}

        # Initialize empty state matrix
        state = np.zeros((MAX_VEHICLES, 5))
        state[:, 4] = 1  # Set all rows as empty initially

        # Collect all vehicles and their properties
        vehicles_data = []

        for direction in directionNumbers.keys():
            for lane in range(3):
                for vehicle in vehicles[direction][lane]:
                    # Get camera position for this direction
                    camera = self.scan_zone_config[direction]["camera"]

                    # Calculate distance to camera based on direction
                    if direction == "right":
                        distance = camera["x"] - (
                            vehicle.x + vehicle.image.get_rect().width
                        )
                    elif direction == "left":
                        distance = vehicle.x - camera["x"]
                    elif direction == "down":
                        distance = camera["y"] - (
                            vehicle.y + vehicle.image.get_rect().height
                        )
                    else:  # up
                        distance = vehicle.y - camera["y"]

                    # Check if vehicle is in scan zone using corner coordinates
                    zone = self.scan_zone_config[direction]["zone"]
                    in_zone = (
                        vehicle.x >= zone["x1"]
                        and vehicle.x <= zone["x2"]
                        and vehicle.y >= zone["y1"]
                        and vehicle.y <= zone["y2"]
                    )

                    # Only include vehicles that are in the scan zone and approaching (positive distance)
                    if in_zone and distance > 0:
                        # Calculate acceleration (1 for accelerating, -1 for decelerating, 0 for constant speed)
                        acceleration = (
                            1
                            if vehicle.accelerated
                            else (-1 if vehicle.decelerated else 0)
                        )

                        vehicles_data.append(
                            {
                                "distance": distance,
                                "speed": vehicle.speed,
                                "acceleration": acceleration,
                                "type": VEHICLE_TYPE_MAP[vehicle.vehicleClass],
                                "direction": direction,
                            }
                        )

        # Sort vehicles by distance (closest first)
        vehicles_data.sort(key=lambda x: x["distance"])

        # Fill state matrix with vehicle data (up to MAX_VEHICLES)
        for i, vehicle in enumerate(vehicles_data[:MAX_VEHICLES]):
            state[i] = [
                vehicle["distance"],
                vehicle["speed"],
                vehicle["acceleration"],
                vehicle["type"],
                0,  # Not empty
            ]

        return state

    def get_state(self):
        """
        Collect the current state of the environment for the RL agent
        Now includes both the original state and the camera state
        """
        # Get original state components
        waiting_count = [0, 0, 0, 0]
        for direction_idx, direction in directionNumbers.items():
            waiting = 0
            for lane in range(3):
                for vehicle in vehicles[direction][lane]:
                    if vehicle.waiting_time > 0:
                        waiting += 1
            waiting_count[direction_idx] = waiting

        # Get current signal state
        signal_state = [0] * 4
        signal_state[currentGreen] = 1 if currentYellow == 0 else 0.5

        # Get queue length in each direction
        queue_lengths = []
        for direction in directionNumbers.values():
            total = 0
            for lane in range(3):
                total += len(vehicles[direction][lane])
            queue_lengths.append(total)

        # Get camera state
        camera_state = self.get_camera_state().flatten()

        # Combine all state components
        state = np.concatenate(
            [waiting_count, signal_state, queue_lengths, [crashes], camera_state]
        )

        # Add to history for time series data
        self.observation_history.append(state)

        # Return observed state
        return state

    def visualize_scan_zone(self, screen):
        """
        Draw the scan zone for each direction on the main simulation screen.
        Uses rectangular zones with corner coordinates.
        """
        # Draw scan zones using corner coordinates
        for direction, config in self.scan_zone_config.items():
            zone = config["zone"]
            # Calculate width and height from corners
            width = zone["x2"] - zone["x1"]
            height = zone["y2"] - zone["y1"]

            # Create surface with calculated dimensions
            scan_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.rect(scan_surface, SCAN_ZONE_COLOR, (0, 0, width, height))
            screen.blit(scan_surface, (zone["x1"], zone["y1"]))

            # Draw camera position (small red dot)
            camera = config["camera"]
            pygame.draw.circle(screen, (255, 0, 0), (camera["x"], camera["y"]), 5)

        # Display cursor coordinates if enabled
        if self.show_coordinates:
            self.display_cursor_coordinates(screen)

    def display_cursor_coordinates(self, screen):
        """
        Display the current cursor coordinates on the screen
        """
        # Get current mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Render coordinate text
        coord_text = f"X: {mouse_x}, Y: {mouse_y}"
        text_surface = self.coordinate_font.render(coord_text, True, (255, 255, 255))

        # Create background for better visibility
        text_width, text_height = text_surface.get_size()
        bg_rect = pygame.Rect(10, 10, text_width + 10, text_height + 10)
        pygame.draw.rect(screen, (0, 0, 0, 200), bg_rect)

        # Draw coordinate text
        screen.blit(text_surface, (15, 15))

    def visualize_camera_state(self):
        """
        Display the camera state in a separate window
        """
        # Clear the camera window
        self.camera_window.fill((0, 0, 0))

        # Get the current camera state
        camera_state = self.get_camera_state()

        # Display header
        header = self.camera_font.render(
            "Camera State Matrix (50 vehicles × 5 attributes)", True, (255, 255, 255)
        )
        self.camera_window.blit(header, (10, 10))

        # Display column headers
        columns = ["Distance", "Speed", "Accel", "Type", "Empty"]
        x_positions = [10, 150, 250, 350, 450]
        for i, col in enumerate(columns):
            text = self.camera_font.render(col, True, (255, 255, 0))
            self.camera_window.blit(text, (x_positions[i], 40))

        # Display vehicle information in a grid format
        y_offset = 70
        for i, vehicle in enumerate(camera_state):
            # Draw row number
            row_num = self.camera_font.render(f"{i+1:2d}", True, (255, 255, 255))
            self.camera_window.blit(row_num, (0, y_offset))

            if vehicle[4] == 0:  # If not empty
                # Format each attribute
                distance = f"{vehicle[0]:.1f}"
                speed = f"{vehicle[1]:.1f}"
                acc_state = ["Const", "Accel", "Decel"][int(vehicle[2]) + 1]
                vehicle_type = ["Car", "Bus", "Truck", "Bike"][int(vehicle[3])]

                # Display each attribute in its column
                texts = [distance, speed, acc_state, vehicle_type, "No"]
                for j, text in enumerate(texts):
                    text_surface = self.camera_font.render(text, True, (255, 255, 255))
                    self.camera_window.blit(text_surface, (x_positions[j], y_offset))
            else:
                # Display empty slot
                text = self.camera_font.render("---", True, (128, 128, 128))
                self.camera_window.blit(text, (x_positions[0], y_offset))
                text = self.camera_font.render("Yes", True, (128, 128, 128))
                self.camera_window.blit(text, (x_positions[4], y_offset))

            y_offset += 25

            # Add a separator line every 10 vehicles
            if (i + 1) % 10 == 0:
                pygame.draw.line(
                    self.camera_window,
                    (128, 128, 128),
                    (0, y_offset),
                    (CAMERA_WINDOW_SIZE[0], y_offset),
                )
                y_offset += 5

        # Update the display
        pygame.display.flip()

        # Print the camera state matrix to console for debugging
        print("\nCamera State Matrix:")
        print("Row | Distance | Speed | Accel | Type  | Empty")
        print("-" * 50)
        for i, vehicle in enumerate(camera_state):
            if vehicle[4] == 0:  # If not empty
                acc_state = ["Const", "Accel", "Decel"][int(vehicle[2]) + 1]
                vehicle_type = ["Car", "Bus", "Truck", "Bike"][int(vehicle[3])]
                print(
                    f"{i+1:3d} | {vehicle[0]:8.1f} | {vehicle[1]:5.1f} | {acc_state:5s} | {vehicle_type:5s} | No"
                )
            else:
                print(
                    f"{i+1:3d} | {'---':8s} | {'---':5s} | {'---':5s} | {'---':5s} | Yes"
                )
        print("-" * 50)

    def step(self, action):
        """
        Take a step in the environment using the specified action
        action: integer 0-3 representing which traffic signal to turn green next
        """
        global currentGreen, nextGreen, crashes, total_reward

        # Action corresponds to setting the next green signal
        nextGreen = action
        logger.debug(
            f"Action taken: {action}, Setting next green signal to {nextGreen}"
        )

        # Run simulation for one time step
        reward = self._calculate_reward()
        self.total_reward += reward
        self.steps += 1

        # Update camera visualization
        self.visualize_camera_state()

        # Get new state
        next_state = self.get_state()

        # Check if done (can be customized based on conditions)
        done = crashes > 0 or self.steps >= 1000

        if done:
            logger.info(
                f"Episode {self.episodes_completed} ended. Steps: {self.steps}, Total Reward: {self.total_reward}, Crashes: {crashes}"
            )

        # Return step information
        return (
            next_state,
            reward,
            done,
            {"crashes": crashes, "total_reward": self.total_reward},
        )

    def _calculate_reward(self):
        """Calculate reward based on waiting times, emissions, and crashes"""
        reward = 0

        # Penalty for waiting vehicles
        for direction in directionNumbers.values():
            for lane in range(3):
                for vehicle in vehicles[direction][lane]:
                    if vehicle.waiting_time > 0:
                        reward += WAITING_PENALTY * min(
                            vehicle.waiting_time, MAX_WAITING_TIME
                        )

        # Penalty for emissions (from braking and accelerating)
        # Use the total emissions that already account for vehicle types
        total_emissions = sum(
            emission_counts[direction]["total"]
            for direction in directionNumbers.values()
        )
        reward += EMISSION_FACTOR * total_emissions

        # Severe penalty for crashes
        if crashes > 0:
            reward += CRASH_PENALTY

        return reward * REWARD_SCALE

    def get_action(self, state):
        """
        Get action using epsilon-greedy policy
        """
        state_key = tuple(state)

        # Initialize Q-values for this state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)

        # Epsilon-greedy action selection
        if random.random() < EPSILON:
            # Explore: choose a random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploit: choose the best action
            return np.argmax(self.q_table[state_key])

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning algorithm
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space)

        # Q-learning update
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])

        # Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        new_q = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_max_q * (1 - done) - current_q
        )
        self.q_table[state_key][action] = new_q
