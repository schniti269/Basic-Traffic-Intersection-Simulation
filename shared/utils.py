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
from enum import Enum
import tqdm

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logger = logging.getLogger("TrafficSimulation")
logger.setLevel(logging.INFO)  # Use INFO level to reduce console output

# Add file handler
timestamp = time.strftime("%Y%m%d-%H%M%S")
file_handler = logging.FileHandler(f"logs/traffic_sim_{timestamp}.log")
file_handler.setLevel(logging.DEBUG)  # Keep DEBUG level for file logging

# Add console handler with reduced verbosity
console_handler = logging.StreamHandler()
console_handler.setLevel(
    logging.WARNING
)  # Further reduce to WARNING to minimize output

# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Performance Mode Flag
PERFORMANCE_MODE = True  # Set to True to reduce logging and show progress bars

# Enable/Disable rendering
ENABLE_RENDERING = False  # Default to False for neural network training
MAX_EPISODES = 100  # Set the maximum number of training episodes for RL
SHOW_FINAL_EPOCH_ONLY = False  # Set to True to only show the final epoch

# Default values of signal timers
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 5

# RL Environment settings
MANUAL_CONTROL = False  # Set to False to use our neural controller
MAX_WAITING_TIME = 100  # Max time a car can wait before maximum penalty
EMISSION_FACTOR = 0.1  # Penalty factor for emissions
CRASH_PENALTY = -1000  # Penalty for a crash
WAITING_PENALTY = -0.1  # Penalty per car waiting per timestep
REWARD_SCALE = 0.01  # Scale factor for rewards

# Q-Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1  # Exploration rate

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

# Vehicle-specific emission factors (CO2 relative units)
emission_factors = {
    "car": 1.0,  # Base emission (1.0 = 100%)
    "bus": 2.5,  # Buses emit 2.5x more than cars
    "truck": 3.0,  # Trucks emit 3x more than cars
    "bike": 0.3,  # Bikes emit 30% of what cars emit
}

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
# Update emission_counts to track by vehicle type as well as direction
emission_counts = {
    "right": {"car": 0, "bus": 0, "truck": 0, "bike": 0, "total": 0},
    "down": {"car": 0, "bus": 0, "truck": 0, "bike": 0, "total": 0},
    "left": {"car": 0, "bus": 0, "truck": 0, "bike": 0, "total": 0},
    "up": {"car": 0, "bus": 0, "truck": 0, "bike": 0, "total": 0},
}
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


# --- Spatial Grid for Collision Detection ---
class SpatialGrid:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_width = (width // cell_size) + 1
        self.grid_height = (height // cell_size) + 1
        self.grid = [[] for _ in range(self.grid_width * self.grid_height)]
        logger.info(
            f"Initialized Spatial Grid: {self.grid_width}x{self.grid_height} cells ({self.cell_size}px size)"
        )

    def _get_cell_indices(self, rect):
        """Get all grid cell indices that a rectangle overlaps."""
        indices = set()
        start_col = max(0, rect.left // self.cell_size)
        end_col = min(self.grid_width - 1, rect.right // self.cell_size)
        start_row = max(0, rect.top // self.cell_size)
        end_row = min(self.grid_height - 1, rect.bottom // self.cell_size)

        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                indices.add(row * self.grid_width + col)
        return indices

    def clear(self):
        """Clear the grid for the next step."""
        for cell in self.grid:
            cell.clear()

    def insert(self, vehicle):
        """Insert a vehicle into the grid based on its current position."""
        # Use the vehicle's current rect for insertion
        rect = pygame.Rect(vehicle.x, vehicle.y, vehicle.width, vehicle.height)
        indices = self._get_cell_indices(rect)
        for index in indices:
            if 0 <= index < len(self.grid):
                self.grid[index].append(vehicle)
            else:
                logger.warning(
                    f"Attempted to insert vehicle {vehicle.id} into invalid grid index {index}."
                )

    def query(self, rect):
        """Query the grid for vehicles potentially colliding with the given rect."""
        potential_colliders = set()
        indices = self._get_cell_indices(rect)
        for index in indices:
            if 0 <= index < len(self.grid):
                potential_colliders.update(self.grid[index])
            else:
                logger.warning(f"Attempted to query invalid grid index {index}.")

        return potential_colliders


# --- End Spatial Grid ---


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
