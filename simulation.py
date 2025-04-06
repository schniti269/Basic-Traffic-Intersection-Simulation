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
        self.accelerated = False
        self.decelerated = False
        self.crashed = False
        vehicles[direction][lane].append(self)

        # Initialize waiting time for this vehicle ID
        if self.id not in waiting_times[direction]:
            waiting_times[direction][self.id] = 0

        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)

        if (
            len(vehicles[direction][lane]) > 1
            and vehicles[direction][lane][self.index - 1].crossed == 0
        ):  # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
            if direction == "right":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].image.get_rect().width
                    - stoppingGap
                )  # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            elif direction == "left":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].image.get_rect().width
                    + stoppingGap
                )
            elif direction == "down":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].image.get_rect().height
                    - stoppingGap
                )
            elif direction == "up":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].image.get_rect().height
                    + stoppingGap
                )
        else:
            self.stop = defaultStop[direction]

        # Set new starting and stopping coordinate
        if direction == "right":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif direction == "left":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif direction == "down":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif direction == "up":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        old_x, old_y = self.x, self.y
        movement_occurred = False

        if self.direction == "right":
            if (
                self.crossed == 0
                and self.x + self.image.get_rect().width > stopLines[self.direction]
            ):  # if the image has crossed stop line now
                self.crossed = 1
            should_move = (
                self.x + self.image.get_rect().width <= self.stop
                or self.crossed == 1
                or (currentGreen == 0 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.x + self.image.get_rect().width
                < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap)
            )
            if should_move:
                self.x += self.speed  # move the vehicle
                movement_occurred = True
        elif self.direction == "down":
            if (
                self.crossed == 0
                and self.y + self.image.get_rect().height > stopLines[self.direction]
            ):
                self.crossed = 1
            should_move = (
                self.y + self.image.get_rect().height <= self.stop
                or self.crossed == 1
                or (currentGreen == 1 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.y + self.image.get_rect().height
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
                    + vehicles[self.direction][self.lane][self.index - 1]
                    .image.get_rect()
                    .width
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
                    + vehicles[self.direction][self.lane][self.index - 1]
                    .image.get_rect()
                    .height
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
            if old_x != self.x or old_y != self.y:
                # Vehicle had to stop (was moving and now stopped)
                self.decelerated = True
                emission_counts[self.direction] += 1
        elif self.waiting_time > 0 and (old_x != self.x or old_y != self.y):
            # Vehicle started moving after waiting
            self.accelerated = True
            emission_counts[self.direction] += 1
            self.waiting_time = 0

        # Check for collision (simplified version)
        if movement_occurred and not self.crashed:
            for vehicle in simulation:
                if vehicle != self and not vehicle.crashed:
                    # Simple rectangular collision detection
                    if (
                        abs(self.x - vehicle.x) < 20
                        and abs(self.y - vehicle.y) < 20
                        and self.direction != vehicle.direction
                    ):
                        global crashes
                        crashes += 1
                        self.crashed = True
                        vehicle.crashed = True
                        return


# RL Environment class
class TrafficEnvironment:
    def __init__(self):
        self.steps = 0
        self.total_reward = 0
        self.episodes_completed = 0
        self.action_space = 4  # Number of possible actions (which signal to turn green)
        self.observation_history = deque(maxlen=5)
        self.q_table = {}  # Q-table for Q-learning
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

        # Create state vector
        state = waiting_count + signal_state + queue_lengths + [crashes]

        # Add to history for time series data
        self.observation_history.append(state)

        # Return observed state
        return np.array(state)

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
        total_emissions = sum(emission_counts.values())
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
    MAX_VEHICLES_PER_SECOND = 5
    vehicle_count = 0
    last_reset_time = time.time()

    logger.info("Vehicle generation started")

    while True:
        # Rate limiting check
        current_time = time.time()
        if current_time - last_reset_time >= 1.0:
            vehicle_count = 0
            last_reset_time = current_time

        # Check if we've reached the rate limit
        if vehicle_count >= MAX_VEHICLES_PER_SECOND:
            time.sleep(0.1)  # Small delay to prevent CPU hogging
            continue

        # Input validation for vehicle parameters
        vehicle_type = random.randint(0, 3)
        lane_number = random.randint(1, 2)
        temp = random.randint(0, 99)
        direction_number = 0
        dist = [25, 50, 75, 100]

        # Validate direction number
        if temp < dist[0]:
            direction_number = 0
        elif temp < dist[1]:
            direction_number = 1
        elif temp < dist[2]:
            direction_number = 2
        elif temp < dist[3]:
            direction_number = 3
        else:
            # This should never happen, but just in case
            direction_number = 0

        # Validate vehicle type and lane number
        if vehicle_type not in vehicleTypes or lane_number not in [1, 2]:
            logger.warning(
                f"Invalid vehicle parameters: type={vehicle_type}, lane={lane_number}"
            )
            continue

        # Create the vehicle
        try:
            Vehicle(
                lane_number,
                vehicleTypes[vehicle_type],
                direction_number,
                directionNumbers[direction_number],
            )
            vehicle_count += 1
            logger.debug(
                f"Vehicle created: {vehicleTypes[vehicle_type]} in {directionNumbers[direction_number]} direction, lane {lane_number}"
            )
        except Exception as e:
            logger.error(f"Error creating vehicle: {e}")

        time.sleep(1)


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
