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
ENABLE_RENDERING = False  # Set to True to see the visualization
MAX_EPISODES = 10  # Set the maximum number of training episodes
SHOW_FINAL_EPOCH_ONLY = True  # Set to True to only show the final epoch

# Default values of signal timers
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 5

# RL Environment settings
MANUAL_CONTROL = False  # Set to False for AI control
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
