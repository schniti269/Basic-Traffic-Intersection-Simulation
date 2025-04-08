import numpy as np
from collections import deque
import random
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
)
from traffic_signal import initialize


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
