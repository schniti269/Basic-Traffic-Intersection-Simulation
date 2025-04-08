import random
import time
import threading
import pygame
import sys
import numpy as np
from collections import deque
import logging
import os
from datetime import datetime
import argparse
import json

# Try to import TensorFlow, but provide fallback to Q-learning if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input
    from tensorflow.keras.optimizers import Adam
    import matplotlib.pyplot as plt

    USE_NEURAL_NETWORK = True
    print("Using TensorFlow Neural Network for learning")
except ImportError:
    USE_NEURAL_NETWORK = False
    print("TensorFlow not available, falling back to Q-learning")

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(
    log_dir, f"traffic_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("TrafficSimulation")

# Enable/Disable rendering
ENABLE_RENDERING = True  # Set to True to see the visualization
MAX_EPISODES = 3  # Set the maximum number of training episodes - reduced for testing
SHOW_FINAL_EPOCH_ONLY = False  # Set to False to show all epochs
STEPS_PER_EPISODE = 1000  # Reduced for testing, set to 100000 for full training
PERFORMANCE_MODE = True  # Set to True to improve performance

# Default values of signal timers
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 5

# RL Environment settings
MANUAL_CONTROL = False  # Set to False for AI control
MAX_WAITING_TIME = 100  # Max time a car can wait before maximum penalty
WAITING_PENALTY_BASE = -0.1  # Base penalty per car waiting per timestep
WAITING_PENALTY_GROWTH_RATE = 0.01  # Rate at which waiting penalty increases
CRASH_PENALTY = -100  # Penalty for a crash
REWARD_SCALE = 0.01  # Scale factor for rewards

# CO2 emission factors for different vehicle types (g/km)
CO2_EMISSIONS = {
    "car": 120,
    "bus": 900,
    "truck": 800,
    "bike": 50,
}

# Activation zone visualization
SHOW_ACTIVATION_ZONES = False  # Set to False to improve performance
ACTIVATION_ZONE_COLOR = (50, 200, 50, 100)  # RGBA format with transparency

# Optimize collision detection
USE_SPATIAL_HASH = True  # Use spatial hashing for faster collision detection
COLLISION_CELL_SIZE = 50  # Size of the spatial hash cells

# Rendering optimization
RENDER_FREQUENCY = 3  # Only render every N frames to improve performance

# Neural Network parameters (only used if TensorFlow is available)
if USE_NEURAL_NETWORK:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9995
    MEMORY_SIZE = 1000  # Reduced from 10000 to improve memory usage
    UPDATE_TARGET_FREQUENCY = 1000

    # Performance optimization for TensorFlow
    try:
        # Use mixed precision to improve performance
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        # DO NOT disable eager execution - it breaks weight transfer between models
        # tf.compat.v1.disable_eager_execution()

        # Set threading options for better CPU utilization
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)

        # Optimize memory usage
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU optimization applied for {len(gpus)} devices")

        logger.info("TensorFlow optimizations applied")
    except Exception as e:
        logger.warning(f"Could not apply TensorFlow optimizations: {e}")
else:
    # Q-Learning parameters (fallback)
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    EPSILON = 0.1  # Exploration rate

# Increase vehicle generation rate but keep it reasonable
MAX_VEHICLES_PER_SECOND = 1  # Increased from 5 to 10 for testing

signals = []
noOfSignals = 4
currentGreen = 0  # Indicates which signal is green currently
nextGreen = (
    currentGreen + 1
) % noOfSignals  # Indicates which signal will turn green next
currentYellow = 0  # Indicates whether yellow signal is on or off

speeds = {
    "car": 2.25,
    "bus": 1.8,
    "truck": 1.8,
    "bike": 2.5,
}  # average speeds of vehicles

# Coordinates of vehicles' start
x = {
    "right": [0, 0, 0],
    "down": [755, 727, 697],
    "left": [1400, 1400, 1400],
    "up": [602, 627, 657],
}
y = {
    "right": [348, 370, 398],
    "down": [0, 0, 0],
    "left": [498, 466, 436],
    "up": [800, 800, 800],
}

vehicles = {
    "right": {0: [], 1: [], 2: [], "crossed": 0},
    "down": {0: [], 1: [], 2: [], "crossed": 0},
    "left": {0: [], 1: [], 2: [], "crossed": 0},
    "up": {0: [], 1: [], 2: [], "crossed": 0},
}
vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}

# Metrics for RL
waiting_vehicles = {"right": 0, "down": 0, "left": 0, "up": 0}
emission_counts = {"right": 0, "down": 0, "left": 0, "up": 0}
crashes = 0
total_reward = 0
waiting_times = {"right": {}, "down": {}, "left": {}, "up": {}}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# Coordinates of stop lines
stopLines = {"right": 590, "down": 330, "left": 800, "up": 535}
defaultStop = {"right": 580, "down": 320, "left": 810, "up": 545}

# Gap between vehicles
stoppingGap = 15  # stopping gap
movingGap = 15  # moving gap

# Initialize Pygame
pygame.init()
simulation = pygame.sprite.Group()


# Add a spatial hash grid for efficient collision detection
class SpatialHash:
    def __init__(self, cell_size=50):
        self.cell_size = cell_size
        self.grid = {}

    def get_cell(self, x, y):
        """Get the cell coordinates for a position"""
        return (int(x / self.cell_size), int(y / self.cell_size))

    def add_object(self, obj):
        """Add an object to the grid"""
        cell = self.get_cell(obj.x, obj.y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(obj)

    def get_nearby_objects(self, obj):
        """Get objects in the same or adjacent cells"""
        cell = self.get_cell(obj.x, obj.y)
        nearby = []

        # Check current cell and adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell[0] + dx, cell[1] + dy)
                if check_cell in self.grid:
                    nearby.extend(self.grid[check_cell])

        return [o for o in nearby if o != obj]

    def clear(self):
        """Clear the grid"""
        self.grid = {}


# Create a global spatial hash
spatial_hash = SpatialHash(COLLISION_CELL_SIZE)


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.id = f"{direction}_{lane}_{len(vehicles[direction][lane])}"
        self.waiting_time = 0
        self.total_waiting_time = 0
        self.accelerated = False
        self.decelerated = False
        self.crashed = False
        self.co2_emitted = 0
        self.in_activation_zone = False
        self.next_check_time = 0  # For collision detection optimization
        vehicles[direction][lane].append(self)

        # Initialize waiting time for this vehicle ID
        if self.id not in waiting_times[direction]:
            waiting_times[direction][self.id] = 0

        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.width = self.image.get_rect().width
        self.height = self.image.get_rect().height

        if (
            len(vehicles[direction][lane]) > 1
            and vehicles[direction][lane][self.index - 1].crossed == 0
        ):  # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
            if direction == "right":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].width
                    - stoppingGap
                )  # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            elif direction == "left":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].width
                    + stoppingGap
                )
            elif direction == "down":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].height
                    - stoppingGap
                )
            elif direction == "up":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].height
                    + stoppingGap
                )
        else:
            self.stop = defaultStop[direction]

        # Set new starting and stopping coordinate
        if direction == "right":
            temp = self.width + stoppingGap
            x[direction][lane] -= temp
        elif direction == "left":
            temp = self.width + stoppingGap
            x[direction][lane] += temp
        elif direction == "down":
            temp = self.height + stoppingGap
            y[direction][lane] -= temp
        elif direction == "up":
            temp = self.height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        # Skip rendering if not in view for performance
        if self.x < -100 or self.x > 1500 or self.y < -100 or self.y > 900:
            return

        screen.blit(self.image, (self.x, self.y))

        # Draw an outline around the vehicle if it's in the activation zone
        if SHOW_ACTIVATION_ZONES and self.in_activation_zone and not PERFORMANCE_MODE:
            rect = self.image.get_rect(topleft=(self.x, self.y))
            zone_surface = pygame.Surface(
                (rect.width + 10, rect.height + 10), pygame.SRCALPHA
            )
            pygame.draw.rect(
                zone_surface,
                ACTIVATION_ZONE_COLOR,
                (0, 0, rect.width + 10, rect.height + 10),
                border_radius=5,
            )
            screen.blit(zone_surface, (rect.x - 5, rect.y - 5))

    def move(self):
        old_x, old_y = self.x, self.y
        movement_occurred = False

        # Check if vehicle is in an activation zone (near intersection)
        if not PERFORMANCE_MODE:
            self.check_activation_zone()

        if self.direction == "right":
            if (
                self.crossed == 0 and self.x + self.width > stopLines[self.direction]
            ):  # if the image has crossed stop line now
                self.crossed = 1
            should_move = (
                self.x + self.width <= self.stop
                or self.crossed == 1
                or (currentGreen == 0 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.x + self.width
                < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap)
            )
            if should_move:
                self.x += self.speed  # move the vehicle
                movement_occurred = True
        elif self.direction == "down":
            if self.crossed == 0 and self.y + self.height > stopLines[self.direction]:
                self.crossed = 1
            should_move = (
                self.y + self.height <= self.stop
                or self.crossed == 1
                or (currentGreen == 1 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.y + self.height
                < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap)
            )
            if should_move:
                self.y += self.speed
                movement_occurred = True
        elif self.direction == "left":
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
            should_move = (
                self.x >= self.stop
                or self.crossed == 1
                or (currentGreen == 2 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.x
                > (
                    vehicles[self.direction][self.lane][self.index - 1].x
                    + vehicles[self.direction][self.lane][self.index - 1].width
                    + movingGap
                )
            )
            if should_move:
                self.x -= self.speed
                movement_occurred = True
        elif self.direction == "up":
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
            should_move = (
                self.y >= self.stop
                or self.crossed == 1
                or (currentGreen == 3 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.y
                > (
                    vehicles[self.direction][self.lane][self.index - 1].y
                    + vehicles[self.direction][self.lane][self.index - 1].height
                    + movingGap
                )
            )
            if should_move:
                self.y -= self.speed
                movement_occurred = True

        # Update metrics for RL
        if not movement_occurred:
            # Vehicle is waiting
            # Ensure the vehicle ID exists in the waiting_times dictionary
            if self.id not in waiting_times[self.direction]:
                waiting_times[self.direction][self.id] = 0

            waiting_times[self.direction][self.id] += 1
            self.waiting_time += 1
            self.total_waiting_time += 1

            if old_x != self.x or old_y != self.y:
                # Vehicle had to stop (was moving and now stopped)
                self.decelerated = True

                if not PERFORMANCE_MODE:
                    # Calculate CO2 emissions based on vehicle type
                    emission_multiplier = 1.5  # Stopping emits more due to inefficiency
                    self.co2_emitted += (
                        CO2_EMISSIONS[self.vehicleClass] * emission_multiplier / 3600
                    )  # g/sec
                    emission_counts[self.direction] += (
                        CO2_EMISSIONS[self.vehicleClass] * emission_multiplier / 3600
                    )
                else:
                    # Simplified emissions for performance mode
                    emission_counts[self.direction] += 1
        elif self.waiting_time > 0 and (old_x != self.x or old_y != self.y):
            # Vehicle started moving after waiting
            self.accelerated = True

            if not PERFORMANCE_MODE:
                # Calculate CO2 emissions based on vehicle type for acceleration
                emission_multiplier = 2.0  # Acceleration emits more
                self.co2_emitted += (
                    CO2_EMISSIONS[self.vehicleClass] * emission_multiplier / 3600
                )  # g/sec
                emission_counts[self.direction] += (
                    CO2_EMISSIONS[self.vehicleClass] * emission_multiplier / 3600
                )
            else:
                # Simplified emissions for performance mode
                emission_counts[self.direction] += 1
            self.waiting_time = 0
        elif not PERFORMANCE_MODE:
            # Vehicle moving normally - only track in regular mode
            self.co2_emitted += CO2_EMISSIONS[self.vehicleClass] / 3600  # g/sec
            emission_counts[self.direction] += CO2_EMISSIONS[self.vehicleClass] / 3600

        # Only check for collisions if we've moved and not too frequently
        current_time = time.time()
        if (
            movement_occurred
            and not self.crashed
            and current_time >= self.next_check_time
        ):
            self.check_collision()
            # Set next check time to throttle collision detection frequency
            self.next_check_time = current_time + 0.1  # Only check every 100ms

        # Add to spatial hash if using it and we've moved
        if USE_SPATIAL_HASH and (old_x != self.x or old_y != self.y):
            spatial_hash.add_object(self)

    def check_collision(self):
        """Optimized collision detection"""
        global crashes

        try:
            if USE_SPATIAL_HASH:
                # Get nearby vehicles using spatial hash - only check a small subset
                nearby_vehicles = spatial_hash.get_nearby_objects(self)

                # Fast check using only approximate positions
                for vehicle in nearby_vehicles:
                    if not vehicle.crashed and self.direction != vehicle.direction:
                        # Simple distance check first (very fast)
                        dx = abs(self.x - vehicle.x)
                        dy = abs(self.y - vehicle.y)

                        # Only proceed to more detailed collision if potentially close
                        if dx < 30 and dy < 30:
                            # More precise hitbox collision detection
                            self_rect = pygame.Rect(
                                self.x, self.y, self.width, self.height
                            )
                            other_rect = pygame.Rect(
                                vehicle.x, vehicle.y, vehicle.width, vehicle.height
                            )

                            if self_rect.colliderect(other_rect):
                                crashes += 1
                                self.crashed = True
                                vehicle.crashed = True
                                return
            else:
                # Faster traditional collision detection using distance-based filtering
                vehicles_to_check = []

                # Only check against vehicles that could possibly collide (different directions)
                for direction in directionNumbers.values():
                    if direction != self.direction:
                        for lane in range(3):
                            vehicles_to_check.extend(vehicles[direction][lane])

                # Check against the filtered list
                for vehicle in vehicles_to_check:
                    if not vehicle.crashed:
                        # Quick distance check first
                        dx = abs(self.x - vehicle.x)
                        dy = abs(self.y - vehicle.y)

                        if dx < 30 and dy < 30:
                            self_rect = pygame.Rect(
                                self.x, self.y, self.width, self.height
                            )
                            other_rect = pygame.Rect(
                                vehicle.x, vehicle.y, vehicle.width, vehicle.height
                            )

                            if self_rect.colliderect(other_rect):
                                crashes += 1
                                self.crashed = True
                                vehicle.crashed = True
                                return
        except Exception as e:
            # Safely handle any errors in collision detection
            logger.error(f"Error in collision detection: {e}")
            # Don't mark as crashed due to error

    def check_activation_zone(self):
        """Check if the vehicle is in an activation zone (near the intersection)"""
        # Define activation zones around the intersection area
        activation_distance = 150  # Distance from stopline to consider "activated"

        if self.direction == "right":
            self.in_activation_zone = (
                stopLines[self.direction] - activation_distance
                < self.x + self.width
                < stopLines[self.direction] + activation_distance
            )
        elif self.direction == "left":
            self.in_activation_zone = (
                stopLines[self.direction] - activation_distance
                < self.x
                < stopLines[self.direction] + activation_distance
            )
        elif self.direction == "down":
            self.in_activation_zone = (
                stopLines[self.direction] - activation_distance
                < self.y + self.height
                < stopLines[self.direction] + activation_distance
            )
        elif self.direction == "up":
            self.in_activation_zone = (
                stopLines[self.direction] - activation_distance
                < self.y
                < stopLines[self.direction] + activation_distance
            )


# RL Environment class
class TrafficEnvironment:
    def __init__(self):
        self.steps = 0
        self.total_reward = 0
        self.episodes_completed = 0
        self.action_space = 4  # Number of possible actions (which signal to turn green)
        self.observation_history = deque(maxlen=5)
        self.state_size = (
            13  # waiting_count(4) + signal_state(4) + queue_lengths(4) + crashes(1)
        )

        # Create activation zone surfaces if visualization is enabled
        self.activation_zones = self._create_activation_zones()

        # Learning mode depends on TensorFlow availability
        if USE_NEURAL_NETWORK:
            # Experience replay buffer
            self.memory = deque(maxlen=MEMORY_SIZE)

            # Neural Network models
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()

            # Epsilon for exploration/exploitation
            self.epsilon = EPSILON_START

            # Training metrics
            self.loss_history = []
            self.reward_history = []
            self.traffic_flow_history = []
            self.waiting_time_history = []

            # Step counter for target network update
            self.step_counter = 0

            logger.info("TrafficEnvironment initialized with neural network")
        else:
            # Q-table for Q-learning (fallback)
            self.q_table = {}
            logger.info("TrafficEnvironment initialized with Q-learning fallback")

    def _build_model(self):
        """Build a neural network model for DQN with error handling"""
        if not USE_NEURAL_NETWORK:
            return None

        try:
            if PERFORMANCE_MODE:
                # Smaller network for better performance
                model = Sequential(
                    [
                        Input(shape=(self.state_size,)),
                        Dense(32, activation="relu"),
                        Dense(self.action_space, activation="linear"),
                    ]
                )

                # Use a simpler optimizer for better performance
                opt = Adam(learning_rate=LEARNING_RATE, epsilon=1e-7)
                model.compile(loss="mse", optimizer=opt, run_eagerly=False)
            else:
                # Full network when performance isn't critical
                model = Sequential(
                    [
                        Input(shape=(self.state_size,)),
                        Dense(64, activation="relu"),
                        Dense(64, activation="relu"),
                        Dense(self.action_space, activation="linear"),
                    ]
                )
                model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

            # Allow for deterministic behavior when not training
            tf.keras.utils.disable_interactive_logging()

            return model
        except Exception as e:
            logger.error(f"Error building neural network: {e}")
            # Fallback to a much simpler model
            logger.info("Falling back to a simpler model architecture")
            model = Sequential(
                [
                    Input(shape=(self.state_size,)),
                    Dense(16, activation="relu"),
                    Dense(self.action_space, activation="linear"),
                ]
            )
            model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE * 2))
            return model

    def update_target_model(self):
        """Update target model weights with current model weights"""
        if not USE_NEURAL_NETWORK:
            return

        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Target model updated")

    def _create_activation_zones(self):
        """Create visualization surfaces for activation zones"""
        zones = {}
        if SHOW_ACTIVATION_ZONES:
            activation_distance = 150
            # Define activation zones for each direction
            for direction in directionNumbers.values():
                if direction == "right":
                    zones[direction] = pygame.Rect(
                        stopLines[direction] - activation_distance,
                        y[direction][1] - 20,
                        activation_distance * 2,
                        100,
                    )
                elif direction == "left":
                    zones[direction] = pygame.Rect(
                        stopLines[direction] - activation_distance,
                        y[direction][1] - 20,
                        activation_distance * 2,
                        100,
                    )
                elif direction == "down":
                    zones[direction] = pygame.Rect(
                        x[direction][1] - 20,
                        stopLines[direction] - activation_distance,
                        100,
                        activation_distance * 2,
                    )
                elif direction == "up":
                    zones[direction] = pygame.Rect(
                        x[direction][1] - 20,
                        stopLines[direction] - activation_distance,
                        100,
                        activation_distance * 2,
                    )
        return zones

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
        emission_counts = {"right": 0, "down": 0, "left": 0, "up": 0}
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

    def get_state(self):
        """
        Collect the current state of the environment for the RL agent
        """
        # Count waiting vehicles at each signal
        waiting_count = [0, 0, 0, 0]
        waiting_by_type = {
            "car": [0, 0, 0, 0],
            "bus": [0, 0, 0, 0],
            "truck": [0, 0, 0, 0],
            "bike": [0, 0, 0, 0],
        }

        for direction_idx, direction in directionNumbers.items():
            waiting = 0
            for lane in range(3):
                for vehicle in vehicles[direction][lane]:
                    if vehicle.waiting_time > 0:
                        waiting += 1
                        waiting_by_type[vehicle.vehicleClass][direction_idx] += 1
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

        # Create state vector
        state = waiting_count + signal_state + queue_lengths + [crashes]

        # Add to history for time series data
        self.observation_history.append(state)

        # Return observed state
        return np.array(state)

    def get_vehicle_type_state(self):
        """Extended state that includes vehicle type information"""
        base_state = self.get_state()

        # Count waiting vehicles by type
        type_counts = []
        for vtype in vehicleTypes.values():
            for direction in directionNumbers.values():
                type_count = 0
                for lane in range(3):
                    for vehicle in vehicles[direction][lane]:
                        if vehicle.vehicleClass == vtype and vehicle.waiting_time > 0:
                            type_count += 1
                type_counts.append(type_count)

        # Combine base state with type information
        return np.concatenate([base_state, np.array(type_counts)])

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

        if USE_NEURAL_NETWORK:
            self.step_counter += 1

        # Get new state
        next_state = self.get_state()

        # Check if done (use episode length for the improvement)
        done = crashes > 0 or self.steps >= STEPS_PER_EPISODE

        # Record metrics for visualization if using neural network
        if USE_NEURAL_NETWORK:
            self.reward_history.append(reward)

            traffic_flow = sum(
                [
                    vehicles[direction]["crossed"]
                    for direction in directionNumbers.values()
                ]
            )
            self.traffic_flow_history.append(traffic_flow)

            avg_waiting_time = 0
            wait_count = 0
            for direction in directionNumbers.values():
                for vehicle_id, wait_time in waiting_times[direction].items():
                    avg_waiting_time += wait_time
                    wait_count += 1
            if wait_count > 0:
                avg_waiting_time /= wait_count
            self.waiting_time_history.append(avg_waiting_time)

        if done:
            logger.info(
                f"Episode {self.episodes_completed} ended. Steps: {self.steps}, Total Reward: {self.total_reward}, Crashes: {crashes}"
            )

            # Plot training metrics if it's the final episode and using neural network
            if USE_NEURAL_NETWORK and self.episodes_completed == MAX_EPISODES:
                self._plot_training_metrics()

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

        # Penalty for waiting vehicles with increasing penalty over time
        for direction in directionNumbers.values():
            for lane in range(3):
                for vehicle in vehicles[direction][lane]:
                    if vehicle.waiting_time > 0:
                        # Increase penalty with waiting time (quadratic growth)
                        waiting_penalty = WAITING_PENALTY_BASE * (
                            1 + WAITING_PENALTY_GROWTH_RATE * vehicle.waiting_time
                        )

                        # Different penalties based on vehicle type
                        if vehicle.vehicleClass == "truck":
                            waiting_penalty *= 3  # Higher penalty for trucks waiting
                        elif vehicle.vehicleClass == "bus":
                            waiting_penalty *= 2.5  # Higher penalty for buses waiting
                        elif vehicle.vehicleClass == "bike":
                            waiting_penalty *= 0.8  # Lower penalty for bikes waiting

                        reward += waiting_penalty * min(
                            vehicle.waiting_time, MAX_WAITING_TIME
                        )

        # Penalty for emissions based on real CO2 output
        total_emissions = sum(emission_counts.values())
        reward -= total_emissions * 0.01  # Scale based on real emissions

        # Severe penalty for crashes
        if crashes > 0:
            reward += CRASH_PENALTY

        return reward * REWARD_SCALE

    def memorize(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        if not USE_NEURAL_NETWORK:
            return

        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the agent (neural network or Q-learning)"""
        if USE_NEURAL_NETWORK:
            return self._train_neural_network()
        else:
            return 0  # Q-learning is handled in update_q_table

    def _train_neural_network(self):
        """Train the neural network with batch from experience replay"""
        if len(self.memory) < BATCH_SIZE:
            return 0

        # Early exit for performance mode if not due for training
        if PERFORMANCE_MODE and self.step_counter % 3 != 0:
            return 0

        # Sample a batch from memory
        minibatch = random.sample(self.memory, BATCH_SIZE)

        # Process batch data more efficiently
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Optimize model prediction
        if PERFORMANCE_MODE:
            # Less accurate but faster training
            targets = np.zeros((BATCH_SIZE, self.action_space))
            next_q_values = self.target_model.predict(
                next_states, verbose=0, batch_size=BATCH_SIZE
            )

            # Vectorized operations for speed
            for i in range(BATCH_SIZE):
                targets[i, actions[i]] = rewards[i] + (
                    1 - dones[i]
                ) * DISCOUNT_FACTOR * np.max(next_q_values[i])
        else:
            # More accurate but slower training
            targets = self.model.predict(states, verbose=0)
            next_q_values = self.target_model.predict(next_states, verbose=0)

            # Update Q values for actions taken
            for i in range(BATCH_SIZE):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + DISCOUNT_FACTOR * np.max(
                        next_q_values[i]
                    )

        # Train the model
        history = self.model.fit(
            states, targets, epochs=1, verbose=0, batch_size=BATCH_SIZE
        )
        loss = history.history["loss"][0]
        self.loss_history.append(loss)

        # Update target model periodically
        if self.step_counter % UPDATE_TARGET_FREQUENCY == 0:
            self.update_target_model()

        # Decay epsilon more aggressively in performance mode
        if PERFORMANCE_MODE:
            if self.epsilon > EPSILON_END:
                self.epsilon *= EPSILON_DECAY * 0.99
        else:
            if self.epsilon > EPSILON_END:
                self.epsilon *= EPSILON_DECAY

        return loss

    def get_action(self, state):
        """
        Get action using epsilon-greedy policy
        """
        if USE_NEURAL_NETWORK:
            return self._get_nn_action(state)
        else:
            return self._get_q_action(state)

    def _get_nn_action(self, state):
        """Get action using epsilon-greedy policy with neural network"""
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploit: choose the best action based on the model prediction
            state_array = np.array([state])  # Convert to batch format for model
            q_values = self.model.predict(state_array, verbose=0)[0]
            return np.argmax(q_values)

    def _get_q_action(self, state):
        """Get action using epsilon-greedy policy with Q-table"""
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
        Update Q-table using Q-learning algorithm (only if not using neural network)
        """
        if USE_NEURAL_NETWORK:
            return  # Neural network doesn't use Q-table

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

    def _plot_training_metrics(self):
        """Plot training metrics at the end of training (neural network only)"""
        if not USE_NEURAL_NETWORK:
            return

        plt.figure(figsize=(15, 10))

        # Plot reward history
        plt.subplot(2, 2, 1)
        plt.plot(self.reward_history)
        plt.title("Reward History")
        plt.xlabel("Step")
        plt.ylabel("Reward")

        # Plot loss history
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history)
        plt.title("Loss History")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")

        # Plot traffic flow
        plt.subplot(2, 2, 3)
        plt.plot(self.traffic_flow_history)
        plt.title("Traffic Flow")
        plt.xlabel("Step")
        plt.ylabel("Total Vehicles Crossed")

        # Plot average waiting time
        plt.subplot(2, 2, 4)
        plt.plot(self.waiting_time_history)
        plt.title("Average Waiting Time")
        plt.xlabel("Step")
        plt.ylabel("Time Steps")

        # Save the figure
        plt.tight_layout()
        plt.savefig("training_metrics.png")
        logger.info("Training metrics saved to training_metrics.png")


# Manual control of signals if not using RL
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(
        ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen[1]
    )
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
    signals.append(ts4)

    if MANUAL_CONTROL:
        repeat()


def repeat():
    global currentGreen, currentYellow, nextGreen
    while (
        signals[currentGreen].green > 0
    ):  # while the timer of current green signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 1  # set yellow signal on
    # reset stop coordinates of lanes and vehicles
    for i in range(0, 3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while (
        signals[currentGreen].yellow > 0
    ):  # while the timer of current yellow signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 0  # set yellow signal off

    # reset all signal times of current signal to default times
    signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed

    currentGreen = nextGreen  # set next signal as green signal
    nextGreen = (currentGreen + 1) % noOfSignals  # set next green signal
    signals[nextGreen].red = (
        signals[currentGreen].yellow + signals[currentGreen].green
    )  # set the red time of next to next signal as (yellow time + green time) of next signal

    if MANUAL_CONTROL:
        repeat()


# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


# Generating vehicles in the simulation
def generateVehicles():
    # Rate limiting - maximum vehicles per second
    MAX_VEHICLES_PER_SECOND = 10  # Increased from 5 to 10 for testing
    vehicle_count = 0
    last_reset_time = time.time()

    # Traffic pattern parameters
    # Rush hour simulation with varying density
    RUSH_HOUR_DENSITY = {
        "morning": {"car": 70, "bus": 10, "truck": 15, "bike": 5},  # Morning rush hour
        "midday": {"car": 40, "bus": 5, "truck": 30, "bike": 25},  # Midday
        "evening": {"car": 75, "bus": 15, "truck": 5, "bike": 5},  # Evening rush hour
        "night": {"car": 60, "bus": 5, "truck": 30, "bike": 5},  # Night time
    }

    # Time-based distribution (cycle through traffic patterns)
    current_pattern = "morning"
    pattern_change_time = time.time() + 60  # Change pattern every minute for testing

    # Direction probability distribution (can be dynamic)
    direction_dist = [25, 25, 25, 25]  # Equal distribution initially

    logger.info("Vehicle generation started with increased capacity")

    while True:
        # Rate limiting check
        current_time = time.time()
        if current_time - last_reset_time >= 1.0:
            vehicle_count = 0
            last_reset_time = current_time

        # Update traffic pattern if needed
        if current_time >= pattern_change_time:
            # Cycle through patterns
            if current_pattern == "morning":
                current_pattern = "midday"
            elif current_pattern == "midday":
                current_pattern = "evening"
            elif current_pattern == "evening":
                current_pattern = "night"
            else:
                current_pattern = "morning"

            # Update direction distribution based on pattern
            if current_pattern == "morning":
                direction_dist = [40, 20, 30, 10]  # More right and left traffic
            elif current_pattern == "evening":
                direction_dist = [30, 10, 40, 20]  # More left and right traffic
            else:
                direction_dist = [25, 25, 25, 25]  # Equal distribution

            pattern_change_time = (
                current_time + 60
            )  # Next change in 1 minute for testing
            logger.info(f"Traffic pattern changed to {current_pattern}")

        # Check if we've reached the rate limit
        if vehicle_count >= MAX_VEHICLES_PER_SECOND:
            time.sleep(0.05)  # Small delay to prevent CPU hogging
            continue

        try:
            # Determine if a vehicle should be generated based on pattern density
            vehicle_probs = list(RUSH_HOUR_DENSITY[current_pattern].values())
            total_prob = sum(vehicle_probs)

            # Normalize probabilities
            vehicle_probs = [p / total_prob * 100 for p in vehicle_probs]

            # Generate vehicle based on probabilities
            rand_val = random.random() * 100
            cumulative = 0

            # Map the random value to a vehicle type
            for i, prob in enumerate(vehicle_probs):
                cumulative += prob
                if rand_val <= cumulative:
                    vehicle_type = i
                    break
            else:
                vehicle_type = 0  # Default to car if something goes wrong

            # Get the vehicle class name
            vehicle_class = vehicleTypes[vehicle_type]

            # Determine lane (0-2)
            lane_number = random.randint(0, 2)

            # Determine direction using the probability distribution
            temp = random.randint(0, 99)
            cum_dist = [direction_dist[0]]
            for i in range(1, len(direction_dist)):
                cum_dist.append(cum_dist[i - 1] + direction_dist[i])

            # Find direction index
            for i, threshold in enumerate(cum_dist):
                if temp < threshold:
                    direction_number = i
                    break
            else:
                direction_number = 0  # Default to right if something goes wrong

            # Create the vehicle with validation
            if (
                0 <= vehicle_type < 4
                and 0 <= lane_number <= 2
                and 0 <= direction_number < 4
            ):
                Vehicle(
                    lane_number,
                    vehicle_class,
                    direction_number,
                    directionNumbers[direction_number],
                )
                vehicle_count += 1
                logger.debug(
                    f"Vehicle created: {vehicle_class} in {directionNumbers[direction_number]} direction, lane {lane_number}"
                )
            else:
                logger.warning(
                    f"Invalid vehicle parameters: type={vehicle_type}, lane={lane_number}, direction={direction_number}"
                )
        except Exception as e:
            logger.error(f"Error creating vehicle: {e}", exc_info=True)

        # Shorter delay for more vehicles
        time.sleep(random.uniform(0.05, 0.15))  # Variable delay


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
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    translucent_green = (0, 255, 0, 128)  # For activation zones

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
        model_type = "NEURAL NETWORK" if USE_NEURAL_NETWORK else "Q-LEARNING"
        pygame.display.set_caption(f"TRAFFIC RL SIMULATION WITH {model_type}")
        logger.info("Pygame display initialized")

        # Loading signal images and font
        redSignal = pygame.image.load("images/signals/red.png")
        yellowSignal = pygame.image.load("images/signals/yellow.png")
        greenSignal = pygame.image.load("images/signals/green.png")
        font = pygame.font.Font(None, 30)

        # Pre-render common text for performance
        if PERFORMANCE_MODE:
            pre_rendered_texts = {}
            common_texts = [
                "Episode: ",
                "Total Reward: ",
                "Crashes: ",
                "Step: ",
                "Waiting Cars: ",
            ]
            for text in common_texts:
                pre_rendered_texts[text] = font.render(text, True, white, black)
    else:
        # Still need font for potential logging/non-visual output if needed later
        pass  # No screen or images needed if not rendering

    thread2 = threading.Thread(
        name="generateVehicles", target=generateVehicles, args=()
    )
    thread2.daemon = True
    thread2.start()
    logger.info("Vehicle generation thread started")

    simulation_step = 0
    render_counter = 0
    clock = pygame.time.Clock()

    # For manual testing of the RL interface
    if not MANUAL_CONTROL:
        # Reset environment
        state = env.reset()
        action = env.get_action(state)  # Get action from agent
        logger.info(
            f"RL {'neural network' if USE_NEURAL_NETWORK else 'Q-learning'} agent initialized"
        )

    # Main simulation loop
    running = True  # Use a flag to control the loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Simulation terminated by user")
                running = False  # Set flag to false to exit loop

        # --- RL Step ---
        if not MANUAL_CONTROL:
            # Take step in environment
            next_state, reward, done, info = env.step(action)

            if USE_NEURAL_NETWORK:
                # Store experience in replay memory
                env.memorize(state, action, reward, next_state, done)

                # Only train every few steps when in performance mode
                if not PERFORMANCE_MODE or env.steps % 5 == 0:
                    # Train the neural network
                    loss = env.train()
            else:
                # Update Q-table
                env.update_q_table(state, action, reward, next_state, done)

            # Get next action
            action = env.get_action(next_state)

            # Update state
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
                        model_type = (
                            "NEURAL NETWORK" if USE_NEURAL_NETWORK else "Q-LEARNING"
                        )
                        pygame.display.set_caption(
                            f"TRAFFIC RL SIMULATION - FINAL EPOCH ({model_type})"
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

                # Clear spatial hash for better performance
                if USE_SPATIAL_HASH:
                    spatial_hash.clear()
        # --- End RL Step ---

        # --- Simulation Update (Movement) ---
        # Update spatial hash if using it
        if USE_SPATIAL_HASH:
            spatial_hash.clear()  # Clear grid before updating

        # This needs to run regardless of rendering
        for vehicle in simulation:
            vehicle.move()  # Move vehicles based on simulation logic
        # --- End Simulation Update ---

        # --- Rendering Section ---
        if ENABLE_RENDERING and (
            not SHOW_FINAL_EPOCH_ONLY or env.episodes_completed >= MAX_EPISODES
        ):
            # Only render every RENDER_FREQUENCY frames in performance mode
            render_counter += 1
            if PERFORMANCE_MODE and render_counter % RENDER_FREQUENCY != 0:
                continue

            screen.blit(background, (0, 0))  # display background in simulation

            # Always initialize signalTexts at the beginning of rendering section
            signalTexts = ["", "", "", ""]

            # Render activation zones if enabled and not in performance mode
            if SHOW_ACTIVATION_ZONES and not PERFORMANCE_MODE:
                activation_zone_surface = pygame.Surface(
                    (screenWidth, screenHeight), pygame.SRCALPHA
                )
                for direction, zone_rect in env.activation_zones.items():
                    pygame.draw.rect(
                        activation_zone_surface, ACTIVATION_ZONE_COLOR, zone_rect
                    )
                screen.blit(activation_zone_surface, (0, 0))

            # Display RL metrics if rendering
            if not MANUAL_CONTROL:
                if PERFORMANCE_MODE:
                    # Simplified metrics display for performance mode
                    metrics_text = [
                        f"Episode: {env.episodes_completed}/{MAX_EPISODES}",
                        f"Total Reward: {env.total_reward:.2f}",
                        f"Crashes: {crashes}",
                        f"Step: {env.steps}/{STEPS_PER_EPISODE}",
                    ]
                else:
                    metrics_text = [
                        f"Episode: {env.episodes_completed}/{MAX_EPISODES}",
                        f"Total Reward: {env.total_reward:.2f}",
                        f"Crashes: {crashes}",
                        f"Step: {env.steps}/{STEPS_PER_EPISODE}",
                        f"Waiting Cars: {sum([len(waiting_times[d]) for d in directionNumbers.values()])}",
                    ]

                    if USE_NEURAL_NETWORK:
                        metrics_text.append(f"Epsilon: {env.epsilon:.4f}")
                    else:
                        metrics_text.append(f"Q-Table Size: {len(env.q_table)}")

                for i, text in enumerate(metrics_text):
                    if PERFORMANCE_MODE and i < len(common_texts):
                        # Use pre-rendered text in performance mode
                        prefix = common_texts[i]
                        value = text[len(prefix) :]

                        screen.blit(pre_rendered_texts[prefix], (10, 10 + i * 30))
                        value_surface = font.render(value, True, white, black)
                        screen.blit(
                            value_surface,
                            (10 + pre_rendered_texts[prefix].get_width(), 10 + i * 30),
                        )
                    else:
                        text_surface = font.render(text, True, white, black)
                        screen.blit(text_surface, (10, 10 + i * 30))

                # Display CO2 emissions if not in performance mode
                if not PERFORMANCE_MODE:
                    total_co2 = sum(emission_counts.values())
                    co2_text = f"CO2 Emissions: {total_co2:.2f} g"
                    co2_surface = font.render(co2_text, True, white, black)
                    screen.blit(co2_surface, (10, 10 + len(metrics_text) * 30))

                    # Display vehicle type counts
                    vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "bike": 0}
                    for direction in directionNumbers.values():
                        for lane in range(3):
                            for vehicle in vehicles[direction][lane]:
                                vehicle_counts[vehicle.vehicleClass] += 1

                    y_offset = 10 + (len(metrics_text) + 1) * 30
                    for vtype, count in vehicle_counts.items():
                        vtype_text = f"{vtype}: {count}"
                        vtype_surface = font.render(vtype_text, True, white, black)
                        screen.blit(vtype_surface, (10, y_offset))
                        y_offset += 30

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

            # Display the vehicles if rendering - use more efficient rendering in performance mode
            if PERFORMANCE_MODE:
                # Only render visible vehicles near the intersection
                visible_vehicles = [
                    v for v in simulation if (400 <= v.x <= 1000 and 200 <= v.y <= 600)
                ]
                for vehicle in visible_vehicles:
                    vehicle.render(screen)
            else:
                for vehicle in simulation:
                    vehicle.render(screen)

            pygame.display.update()
        # --- End Rendering Section ---

        # Control simulation speed - increase speed during training
        if not ENABLE_RENDERING or env.episodes_completed < MAX_EPISODES:
            clock.tick(
                2000 if PERFORMANCE_MODE else 1000
            )  # Run even faster in performance mode
        else:
            clock.tick(60)  # Normal speed for visualization

    # Clean up Pygame if it was initialized
    if ENABLE_RENDERING:
        pygame.quit()

    # Save the trained model if using neural network
    if (
        not MANUAL_CONTROL
        and USE_NEURAL_NETWORK
        and env.episodes_completed >= MAX_EPISODES
    ):
        try:
            env.model.save("traffic_nn_model.h5")
            logger.info("Neural network model saved to traffic_nn_model.h5")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    sys.exit()  # Exit after loop finishes


# For direct execution as a script
if __name__ == "__main__":
    main()
