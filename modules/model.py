import numpy as np
import random
import logging
from collections import deque
import time
import json
import os
from datetime import datetime

# Try to import TensorFlow, but provide fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    import matplotlib.pyplot as plt

    USE_NEURAL_NETWORK = True
    print("Using TensorFlow Neural Network for learning")
except ImportError:
    USE_NEURAL_NETWORK = False
    print("TensorFlow not available, falling back to Q-learning")

logger = logging.getLogger("TrafficModel")

# Neural Network parameters
if USE_NEURAL_NETWORK:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9995
    MEMORY_SIZE = 1000
    UPDATE_TARGET_FREQUENCY = 1000

    # Performance optimization for TensorFlow
    try:
        # Use mixed precision to improve performance
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

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

# RL parameters
PERFORMANCE_MODE = True
WAITING_PENALTY_BASE = -0.1  # Base penalty per car waiting per timestep
WAITING_PENALTY_GROWTH_RATE = 0.01  # Rate at which waiting penalty increases
CRASH_PENALTY = -100  # Penalty for a crash
REWARD_SCALE = 0.01  # Scale factor for rewards
MAX_WAITING_TIME = 100  # Maximum waiting time to consider for penalties


class TrafficAgent:
    """RL agent for traffic control"""

    def __init__(self, state_size=13, action_space=4):
        self.steps = 0
        self.total_reward = 0
        self.episodes_completed = 0
        self.action_space = action_space
        self.state_size = state_size
        self.observation_history = deque(maxlen=5)

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

            logger.info("TrafficAgent initialized with neural network")
        else:
            # Q-table for Q-learning (fallback)
            self.q_table = {}
            logger.info("TrafficAgent initialized with Q-learning fallback")

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

    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
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

    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning algorithm (only if not using neural network)"""
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

    def calculate_reward(self, state_info):
        """Calculate reward based on waiting times, emissions, and crashes"""
        reward = 0

        # Extract state information
        waiting_vehicles = state_info.get("waiting_vehicles", {})
        emission_counts = state_info.get("emission_counts", {})
        crashes = state_info.get("crashes", 0)

        # Penalty for waiting vehicles with increasing penalty over time
        for vehicle_type, count in waiting_vehicles.items():
            # Different penalties based on vehicle type
            waiting_penalty = WAITING_PENALTY_BASE

            if vehicle_type == "truck":
                waiting_penalty *= 3  # Higher penalty for trucks waiting
            elif vehicle_type == "bus":
                waiting_penalty *= 2.5  # Higher penalty for buses waiting
            elif vehicle_type == "bike":
                waiting_penalty *= 0.8  # Lower penalty for bikes waiting

            # Apply penalty based on count * waiting time
            reward += waiting_penalty * count

        # Penalty for emissions based on real CO2 output
        total_emissions = sum(emission_counts.values())
        reward -= total_emissions * 0.01  # Scale based on real emissions

        # Severe penalty for crashes
        if crashes > 0:
            reward += CRASH_PENALTY

        return reward * REWARD_SCALE

    def plot_training_metrics(self, filename="training_metrics.png"):
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
        plt.savefig(filename)
        logger.info(f"Training metrics saved to {filename}")

    def save_model(self, filename="traffic_nn_model.h5"):
        """Save the neural network model to file"""
        if not USE_NEURAL_NETWORK:
            logger.warning("Cannot save model: TensorFlow not available")
            return

        try:
            self.model.save(filename)
            logger.info(f"Neural network model saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, filename="traffic_nn_model.h5"):
        """Load a neural network model from file"""
        if not USE_NEURAL_NETWORK:
            logger.warning("Cannot load model: TensorFlow not available")
            return False

        try:
            self.model = tf.keras.models.load_model(filename)
            self.update_target_model()
            logger.info(f"Neural network model loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class TrafficModel:
    """Base class for traffic optimization models"""

    def __init__(self, name="BaseModel"):
        """Initialize the model"""
        self.name = name

    def decide_action(self, state):
        """Decide on an action based on the current state"""
        raise NotImplementedError("Subclasses must implement decide_action")

    def train(self, state, action, reward, next_state):
        """Train the model on a single experience"""
        raise NotImplementedError("Subclasses must implement train")

    def save(self, filepath):
        """Save the model to a file"""
        raise NotImplementedError("Subclasses must implement save")

    def load(self, filepath):
        """Load the model from a file"""
        raise NotImplementedError("Subclasses must implement load")


class RandomModel(TrafficModel):
    """Model that makes random decisions"""

    def __init__(self):
        """Initialize the random model"""
        super().__init__(name="RandomModel")

    def decide_action(self, state):
        """Randomly decide which traffic direction should have green lights"""
        # 0: north-south green, east-west red
        # 1: north-south red, east-west green
        return random.randint(0, 1)

    def train(self, state, action, reward, next_state):
        """Random model doesn't learn"""
        pass

    def save(self, filepath):
        """Nothing to save for random model"""
        logger.info("Random model has no parameters to save")

    def load(self, filepath):
        """Nothing to load for random model"""
        logger.info("Random model has no parameters to load")


class FixedTimeModel(TrafficModel):
    """Model that alternates traffic lights based on a fixed time schedule"""

    def __init__(self, cycle_duration=100):
        """Initialize the fixed time model"""
        super().__init__(name="FixedTimeModel")
        self.cycle_duration = cycle_duration
        self.current_step = 0

    def decide_action(self, state):
        """Alternate traffic lights based on fixed schedule"""
        self.current_step = (self.current_step + 1) % self.cycle_duration

        # First half of cycle: north-south green (0)
        # Second half of cycle: east-west green (1)
        if self.current_step < self.cycle_duration // 2:
            return 0
        else:
            return 1

    def train(self, state, action, reward, next_state):
        """Fixed model doesn't learn"""
        pass

    def save(self, filepath):
        """Save fixed model parameters"""
        with open(filepath, "w") as f:
            json.dump(
                {
                    "name": self.name,
                    "cycle_duration": self.cycle_duration,
                    "current_step": self.current_step,
                },
                f,
            )
        logger.info(f"Fixed time model saved to {filepath}")

    def load(self, filepath):
        """Load fixed model parameters"""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                self.cycle_duration = data.get("cycle_duration", 100)
                self.current_step = data.get("current_step", 0)
            logger.info(f"Fixed time model loaded from {filepath}")
        else:
            logger.warning(f"Model file {filepath} not found")


class QlearningModel(TrafficModel):
    """Model that uses Q-learning to optimize traffic control"""

    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        """Initialize the Q-learning model"""
        super().__init__(name="QlearningModel")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}  # State-action value function
        self.state_encoder = StateEncoder()

    def decide_action(self, state):
        """Decide action using epsilon-greedy policy based on Q-values"""
        # Encode state
        encoded_state = self.state_encoder.encode(state)

        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.randint(0, 1)

        # Exploitation: best action based on Q-values
        return self._get_best_action(encoded_state)

    def _get_best_action(self, encoded_state):
        """Get the best action for a state based on Q-values"""
        # Initialize state in Q-table if not present
        if encoded_state not in self.q_table:
            self.q_table[encoded_state] = [0, 0]  # Q-values for actions 0 and 1

        # Find action with highest Q-value
        q_values = self.q_table[encoded_state]

        # Handle ties randomly
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]

        return random.choice(best_actions)

    def train(self, state, action, reward, next_state):
        """Update Q-values based on the experience"""
        # Encode states
        encoded_state = self.state_encoder.encode(state)
        encoded_next_state = self.state_encoder.encode(next_state)

        # Initialize states in Q-table if not present
        if encoded_state not in self.q_table:
            self.q_table[encoded_state] = [0, 0]

        if encoded_next_state not in self.q_table:
            self.q_table[encoded_next_state] = [0, 0]

        # Calculate target Q-value using the Bellman equation
        best_next_action = self._get_best_action(encoded_next_state)
        target_q = (
            reward
            + self.discount_factor * self.q_table[encoded_next_state][best_next_action]
        )

        # Update Q-value for the current state-action pair
        current_q = self.q_table[encoded_state][action]
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[encoded_state][action] = new_q

    def save(self, filepath):
        """Save the Q-table and parameters to a file"""
        data = {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "q_table": {
                str(k): v for k, v in self.q_table.items()
            },  # Convert tuple keys to strings
        }

        with open(filepath, "w") as f:
            json.dump(data, f)

        logger.info(f"Q-learning model saved to {filepath}")

    def load(self, filepath):
        """Load the Q-table and parameters from a file"""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)

                self.learning_rate = data.get("learning_rate", 0.1)
                self.discount_factor = data.get("discount_factor", 0.95)
                self.exploration_rate = data.get("exploration_rate", 0.1)

                # Convert string keys back to tuples
                q_table_data = data.get("q_table", {})
                self.q_table = {}
                for k, v in q_table_data.items():
                    self.q_table[eval(k)] = v

            logger.info(f"Q-learning model loaded from {filepath}")
        else:
            logger.warning(f"Model file {filepath} not found")


class StateEncoder:
    """Utility class to encode traffic state into a format usable by the RL model"""

    def encode(self, state):
        """
        Encode the traffic state into a tuple that can be used as a dictionary key.

        Args:
            state: A dictionary containing traffic state information
                - current_phase: 0 (N-S green) or 1 (E-W green)
                - waiting_vehicles: Count of waiting vehicles in each direction
                - queue_lengths: Length of vehicle queues in each direction

        Returns:
            A tuple representation of the state
        """
        # Extract key features from state
        current_phase = state.get("current_phase", 0)

        # Count waiting vehicles in each direction
        waiting_ns = state.get("waiting_vehicles", {}).get("north", 0) + state.get(
            "waiting_vehicles", {}
        ).get("south", 0)
        waiting_ew = state.get("waiting_vehicles", {}).get("east", 0) + state.get(
            "waiting_vehicles", {}
        ).get("west", 0)

        # Discretize waiting vehicles count to reduce state space
        waiting_ns_disc = min(waiting_ns // 2, 5)  # Max 5 buckets
        waiting_ew_disc = min(waiting_ew // 2, 5)  # Max 5 buckets

        # Create a tuple that represents the state
        encoded_state = (current_phase, waiting_ns_disc, waiting_ew_disc)

        return encoded_state


class RewardCalculator:
    """Utility class to calculate rewards for the RL model"""

    def calculate_reward(self, state, action, next_state):
        """
        Calculate the reward for taking an action in a state and transitioning to next_state.

        Args:
            state: State before the action
            action: The action taken (0 or 1)
            next_state: State after the action

        Returns:
            A numerical reward value
        """
        # Calculate reward based on total waiting time
        total_waiting_before = self._get_total_waiting(state)
        total_waiting_after = self._get_total_waiting(next_state)

        # Reward is negative waiting time difference (less waiting = positive reward)
        waiting_reward = -(total_waiting_after - total_waiting_before)

        # Penalize switching phases too frequently
        switching_penalty = 0
        if state.get("current_phase") != action:
            switching_penalty = -1

        # Reward for good throughput
        throughput_reward = next_state.get("processed_vehicles", 0) - state.get(
            "processed_vehicles", 0
        )

        # Combine rewards with different weights
        total_reward = (
            0.7 * waiting_reward + 0.2 * switching_penalty + 0.1 * throughput_reward
        )

        return total_reward

    def _get_total_waiting(self, state):
        """Helper to get the total waiting time from a state"""
        waiting_vehicles = state.get("waiting_vehicles", {})
        return sum(waiting_vehicles.values())


def create_model(model_type="fixed", **kwargs):
    """Factory function to create traffic models"""
    if model_type.lower() == "random":
        return RandomModel()
    elif model_type.lower() == "fixed":
        cycle_duration = kwargs.get("cycle_duration", 100)
        return FixedTimeModel(cycle_duration=cycle_duration)
    elif model_type.lower() == "qlearning":
        learning_rate = kwargs.get("learning_rate", 0.1)
        discount_factor = kwargs.get("discount_factor", 0.95)
        exploration_rate = kwargs.get("exploration_rate", 0.1)
        return QlearningModel(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
        )
    else:
        logger.warning(
            f"Unknown model type: {model_type}. Using FixedTimeModel as default."
        )
        return FixedTimeModel()
