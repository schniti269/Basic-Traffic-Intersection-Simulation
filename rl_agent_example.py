import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
from simulation import TrafficEnvironment


class QLearningAgent:
    """
    Simple Q-learning agent for traffic signal control
    """

    def __init__(
        self,
        action_space=4,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration=0.01,
    ):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # For this simple example, we'll use a state discretization approach
        # to create a manageable Q-table
        self.q_table = {}

    def discretize_state(self, state):
        """
        Convert continuous state to discrete representation for Q-table
        This is a simplified approach - in practice you'd want a more sophisticated solution
        """
        # Extract key metrics from state
        # First 4 values are waiting cars in each direction
        waiting_cars = state[:4]

        # Discretize waiting cars into buckets: 0, 1-5, 6-10, >10
        discrete_waiting = tuple(
            [
                0 if count == 0 else 1 if count <= 5 else 2 if count <= 10 else 3
                for count in waiting_cars
            ]
        )

        # Add current green signal (1-hot encoded in positions 4-7)
        current_signal = np.argmax(state[4:8]) if np.max(state[4:8]) > 0 else 0

        # Create a hashable state representation
        discrete_state = discrete_waiting + (current_signal,)
        return discrete_state

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy
        """
        discrete_state = self.discretize_state(state)

        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space - 1)

        # Exploitation: best action from Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space)

        return np.argmax(self.q_table[discrete_state])

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-learning update rule
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        # Initialize Q-values if they don't exist
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_space)

        # Q-learning update
        current_q = self.q_table[discrete_state][action]

        # Terminal state handling
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])

        # Q-value update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[discrete_state][action] = new_q

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration, self.exploration_rate * self.exploration_decay
        )

    def save_q_table(self, filename="q_table.npy"):
        """Save the Q-table to a file"""
        # Convert dictionary to a serializable format
        keys = list(self.q_table.keys())
        values = [self.q_table[k] for k in keys]
        np.save(filename, {"keys": keys, "values": values})

    def load_q_table(self, filename="q_table.npy"):
        """Load the Q-table from a file"""
        data = np.load(filename, allow_pickle=True).item()
        self.q_table = {tuple(k): v for k, v in zip(data["keys"], data["values"])}


def train_agent(num_episodes=100, max_steps=500, render_frequency=10):
    """
    Train the Q-learning agent on the traffic environment
    """
    env = TrafficEnvironment()
    agent = QLearningAgent()

    rewards_history = []
    crashes_history = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            # Select action
            action = agent.choose_action(state)

            # Take action in environment
            next_state, reward, done, info = env.step(action)

            # Update Q-values
            agent.learn(state, action, reward, next_state, done)

            # Update state and metrics
            state = next_state
            total_reward += reward
            step += 1

            # Render every N episodes
            if episode % render_frequency == 0:
                pygame.display.update()

        # Record metrics
        rewards_history.append(total_reward)
        crashes_history.append(info["crashes"])

        print(
            f"Episode {episode+1}/{num_episodes}, "
            f"Steps: {step}, Total Reward: {total_reward:.2f}, "
            f"Crashes: {info['crashes']}, "
            f"Exploration: {agent.exploration_rate:.2f}"
        )

    # Save trained agent
    agent.save_q_table()

    # Plot learning curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot(crashes_history)
    plt.title("Crashes per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Number of Crashes")

    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()

    return agent


def test_trained_agent(agent=None, num_episodes=5, max_steps=300):
    """
    Test a trained agent or load one from a file
    """
    env = TrafficEnvironment()

    if agent is None:
        agent = QLearningAgent()
        try:
            agent.load_q_table()
            print("Loaded pre-trained agent")
        except:
            print("No pre-trained agent found, using random actions")

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            # Always choose best action during testing (no exploration)
            action = np.argmax(
                agent.q_table.get(
                    agent.discretize_state(state), np.zeros(agent.action_space)
                )
            )

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update state and metrics
            state = next_state
            total_reward += reward
            step += 1

            # Render
            pygame.display.update()

        print(
            f"Test Episode {episode+1}, Steps: {step}, Total Reward: {total_reward:.2f}"
        )


if __name__ == "__main__":
    print("Traffic Intersection RL with Q-Learning")
    print("--------------------------------------")

    # Ask user whether to train or test
    mode = input("Choose mode (train/test): ").strip().lower()

    if mode == "train":
        num_episodes = int(input("Number of episodes (default 50): ") or "50")
        agent = train_agent(num_episodes=num_episodes)

        # Ask if user wants to test the trained agent
        if input("Test the trained agent? (y/n): ").strip().lower() == "y":
            test_trained_agent(agent)

    elif mode == "test":
        # Load and test a pre-trained agent
        test_trained_agent()

    else:
        print("Invalid mode. Please choose 'train' or 'test'.")
