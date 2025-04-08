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

# Enable/Disable rendering
ENABLE_RENDERING = True  # Set to True to see the visualization
MAX_EPISODES = 10  # Set the maximum number of training episodes
SHOW_FINAL_EPOCH_ONLY = False  # Set to True to only show the final epoch

# Default values of signal timers
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 5

# RL Environment settings
MANUAL_CONTROL = True  # Set to True for manual control (no AI)
MAX_WAITING_TIME = 100  # Max time a car can wait before maximum penalty
EMISSION_FACTOR = 0.1  # Penalty factor for emissions
CRASH_PENALTY = -100  # Penalty for a crash
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
