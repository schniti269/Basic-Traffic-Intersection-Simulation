# Neural Network Traffic Light Controller

This project implements a neural network-based traffic light controller for a traffic intersection simulation. The neural network processes data from vehicle detection zones and makes decisions about which traffic light should be green.

## Features

- Deep neural network for traffic light control
- Reinforcement learning approach with memory replay
- Automatic training with visual feedback every 10 epochs
- Reward system based on:
  - Average vehicle speed (higher is better)
  - Crash prevention
  - Minimized waiting time
  - Reduced CO2 emissions based on vehicle type

## Files

- `main.py`: Main simulation file that runs the traffic simulation
- `neural_model_01.py`: Neural network model for traffic light control
- `utils.py`: Utility functions, constants, and global variables
- `vehicle.py`: Vehicle class definition
- `traffic_signal.py`: Traffic signal control logic
- `rl_environment.py`: Original reinforcement learning environment (Q-learning)

## How to Run

```bash
# Install required packages
pip install pygame tensorflow numpy

# Run the simulation
python main.py
```

## How it Works

1. The simulation detects vehicles in four scan zones (right, down, left, up)
2. Vehicle data is processed into a state matrix for the neural network
3. The neural network outputs an action (which traffic light should be green)
4. Reward is calculated based on:
   - Average vehicle speed
   - Number of crashes (heavily penalized)
   - Number of waiting vehicles
   - Total CO2 emissions based on vehicle type
5. Experience is stored in memory for training
6. After 1000 simulation steps, the model trains on a batch of experiences
7. Every 10 epochs, the simulation is visualized until the user presses the space key

## Neural Network Architecture

The neural network architecture consists of:
- Input: 4×20×5 matrix (4 zones, 20 vehicles per zone, 5 features per vehicle)
- Hidden layer 1: 256 units with ReLU activation
- Hidden layer 2: 128 units with ReLU activation
- Output: 4 units with sigmoid activation (one for each traffic light)

## Controls

- Press keys 1-4 to display vehicle matrices for each zone in the console
- Press the space key to continue training after visualization
- Press 'C' to toggle coordinate display

## Customization

You can adjust the neural network parameters in `neural_model_01.py`:
- Learning rate
- Epsilon (exploration rate)
- Gamma (discount factor)
- Batch size
- Memory size
- Training interval
- Render interval

## Future Work

- Implement more sophisticated neural architectures (CNN, LSTM)
- Add more features to the state representation
- Optimize reward function
- Implement traffic generation patterns
- Compare performance against traditional traffic control methods 