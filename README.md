<p align="center">
 <img height=350px src="./simulation-output.png" alt="Simulation output">
</p>

<h1 align="center">Traffic Intersection RL Environment</h1>

<div align="center">

[![Python version](https://img.shields.io/badge/python-3.1+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<h4>A traffic intersection simulation refactored as a reinforcement learning environment for traffic signal control optimization.</h4>

</div>

-----------------------------------------
### Description

* A 4-way traffic intersection with traffic signals that can be controlled by an AI agent
* RL interface that rewards optimal traffic flow and penalizes:
  - Traffic accidents/crashes
  - Vehicle waiting time (cars standing at signals)
  - Emissions from unnecessary braking and acceleration
* Vehicles such as cars, bikes, buses, and trucks are generated, and their movement is controlled according to AI-controlled signals
* Provides standard RL interface with state observations, actions, and rewards

This is based on the original [Basic Traffic Intersection Simulation](https://github.com/mihir-m-gandhi/Basic-Traffic-Intersection-Simulation) by Mihir Gandhi.

------------------------------------------
### Reinforcement Learning Environment

The simulation has been refactored to create a reinforcement learning environment with:

#### State Representation
* Number of waiting vehicles at each signal
* Current signal states
* Queue lengths in each direction
* Crash status

#### Actions
* Control which traffic signal turns green next (0-3)

#### Rewards
* Negative reward for vehicles waiting at signals
* Negative reward for emissions from braking and accelerating
* Large negative penalty for crashes

------------------------------------------
### Prerequisites

* [Python 3.1+](https://www.python.org/downloads/)
* NumPy
* PyGame

------------------------------------------
### Installation

 * Step I: Clone the Repository
```sh
      $ git clone https://github.com/yourusername/Traffic-Intersection-RL-Environment
```
  * Step II: Install the required packages
```sh
      # On the terminal, move into Traffic-Intersection-RL-Environment directory
      $ cd Traffic-Intersection-RL-Environment
      $ pip install pygame numpy
```
* Step III: Run the code
```sh
      # To run simulation
      $ python simulation.py
```

------------------------------------------
### Using as an RL Environment

The simulation includes a `TrafficEnvironment` class that follows standard RL conventions:

```python
# Import the environment
from simulation import TrafficEnvironment

# Create and reset the environment
env = TrafficEnvironment()
state = env.reset()

# Run a training loop
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Replace with your agent's policy
        action = your_agent.choose_action(state)
        
        # Take an action in the environment
        next_state, reward, done, info = env.step(action)
        
        # Train your agent
        your_agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode}, Total Reward: {total_reward}")
```

------------------------------------------
### Manual Testing

You can test the environment manually:
- Press keys 0-3 to set which signal turns green next
- Watch the metrics shown on the screen to see how your decisions affect traffic flow

------------------------------------------
### License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
