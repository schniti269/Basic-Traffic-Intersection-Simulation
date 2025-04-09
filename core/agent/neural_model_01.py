import numpy as np
import pygame
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.utils import PERFORMANCE_MODE
import logging

# Default scan zone config (changed to dictionary)
DEFAULT_SCAN_ZONE_CONFIG = {
    "right": {
        "zone": {
            "x1": 811,  # Left edge
            "y1": 427,  # Top edge
            "x2": 1400,  # Right edge
            "y2": 512,  # Bottom edge
        },
        "camera": {
            "x": 787,  # Camera x position
            "y": 464,  # Camera y position
        },
    },
    "left": {
        "zone": {
            "x1": 0,  # Left edge
            "y1": 370,  # Top edge
            "x2": 580,  # Right edge
            "y2": 424,  # Bottom edge
        },
        "camera": {
            "x": 600,  # Camera x position
            "y": 400,  # Camera y position
        },
    },
    "down": {
        "zone": {
            "x1": 600,  # Left edge
            "y1": 546,  # Top edge
            "x2": 681,  # Right edge
            "y2": 800,  # Bottom edge
        },
        "camera": {
            "x": 730,  # Camera x position
            "y": 330,  # Camera y position
        },
    },
    "up": {
        "zone": {
            "x1": 688,  # Left edge
            "y1": 0,  # Top edge
            "x2": 767,  # Right edge
            "y2": 321,  # Bottom edge
        },
        "camera": {
            "x": 650,  # Camera x position
            "y": 310,  # Camera y position
        },
    },
}


logger = logging.getLogger(__name__)


class NeuralTrafficController:
    def __init__(self, steps_per_epoch=1000, total_epochs=100):
        # Configuration
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.input_shape = (
            4,
            20,
            5,
        )  # 4 zones, max 20 vehicles per zone, 5 features per vehicle
        self.output_size = 4  # 4 traffic lights
        self.learning_rate = 0.001
        self.discount_factor = 0.99  # Gamma for discounted returns
        self.render_interval = 10  # epochs between renders

        # State tracking for epoch-based training (REINFORCE)
        self.epoch_states = []
        # self.epoch_actions = [] # Store the raw probabilities output by the model - No longer needed directly for training
        self.epoch_actual_actions = []  # Store the thresholded actions taken
        self.epoch_rewards = []  # Store per-step rewards
        self.epoch_log_probs = []  # Store log probabilities of actions taken

        # Standard state tracking
        self.total_steps = 0
        self.current_epoch = 0
        self.epoch_accumulated_reward = 0  # Accumulates step rewards for reporting
        self.epoch_steps = 0
        self.show_render = False
        self.waiting_for_space = False

        # Aggregated metrics for end-of-epoch summary (reset each epoch)
        self.total_crashes_epoch = 0
        self.total_waiting_steps_epoch = 0  # Sum of waiting vehicles each step
        self.total_emissions_epoch = 0
        self.sum_avg_speeds_epoch = 0  # Sum of avg speeds to calculate overall average
        self.vehicle_updates_epoch = 0  # Count how many times avg speed was calculated

        # Initialize the model and optimizer
        try:
            if not PERFORMANCE_MODE:
                print("Initializing neural network model...")
            self.model = self._build_model()
            # Store the optimizer instance
            self.optimizer = Adam(learning_rate=self.learning_rate)
            if not PERFORMANCE_MODE:
                print("Neural network model initialized successfully")
        except Exception as e:
            print(f"ERROR initializing neural network: {e}")
            print("Using random action fallback instead of neural network")
            self.model = None
            self.optimizer = None

    def _build_model(self):
        """Build a simple neural network model"""
        # No change needed here, still outputs sigmoid probabilities
        try:
            model = Sequential(
                [
                    Flatten(input_shape=self.input_shape),
                    Dense(256, activation="relu"),
                    Dense(128, activation="relu"),
                    Dense(
                        self.output_size, activation="sigmoid"
                    ),  # Outputs probs [0, 1]
                ]
            )
            # Compile is not strictly necessary when using GradientTape, but doesn't hurt
            # model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy') # Example
            return model
        except Exception as e:
            print(f"Error building neural network model: {e}")
            raise e

    def get_state(self, scan_zones):
        """
        Convert scan zone data to a state representation

        Parameters:
        - scan_zones: List of 4 lists containing vehicle data from each zone

        Returns:
        - state matrix of shape (4, 20, 5)
        """
        state = np.zeros(self.input_shape)

        for zone_idx, zone_vehicles in enumerate(scan_zones):
            # Sort vehicles by distance
            zone_vehicles.sort(key=lambda x: x["distance"])

            # Take up to 20 vehicles per zone
            for vehicle_idx, vehicle_data in enumerate(zone_vehicles[:20]):
                if vehicle_idx >= 20:
                    break

                # Extract features:
                # 1. Distance (normalized)
                # 2. Speed (normalized)
                # 3. Vehicle type encoded (car=0, bus=0.33, truck=0.67, bike=1)
                # 4. Acceleration (-1, 0, 1)
                # 5. Waiting time (normalized)

                vehicle = vehicle_data["vehicle"]
                vehicle_type_map = {"car": 0, "bus": 0.33, "truck": 0.67, "bike": 1}

                state[zone_idx, vehicle_idx, 0] = min(
                    vehicle_data["distance"] / 500, 1.0
                )
                state[zone_idx, vehicle_idx, 1] = min(vehicle_data["speed"] / 10, 1.0)
                state[zone_idx, vehicle_idx, 2] = vehicle_type_map[vehicle_data["type"]]
                state[zone_idx, vehicle_idx, 3] = (
                    vehicle_data["acceleration"] + 1
                ) / 2  # Map [-1,1] to [0,1]
                state[zone_idx, vehicle_idx, 4] = min(vehicle.waiting_time / 100, 1.0)

        return state

    def get_action(self, state):
        """
        Get traffic light action probabilities and thresholded actions.
        Returns:
        - tuple: (list of booleans for active lights, raw probabilities np.array)
        """
        try:
            if self.model is None:
                # Fallback: Randomly activate 1 or 2 lights
                num_active = random.randint(1, 2)
                action_indices = random.sample(range(self.output_size), num_active)
                active_lights = [i in action_indices for i in range(self.output_size)]
                # Return dummy probabilities for consistency
                dummy_probs = np.array(
                    [0.5 if active else 0.0 for active in active_lights]
                )
                return active_lights, dummy_probs

            # Predict probabilities for each light
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            action_probs = self.model(state_tensor, training=False)[0].numpy()
            # No need for print statements here due to PERFORMANCE_MODE check in update

            # Apply threshold to determine active lights
            active_lights = (action_probs >= 0.5).tolist()

            return active_lights, action_probs

        except Exception as e:
            print(f"Error getting action: {e}")
            # Fallback: Randomly activate 1 or 2 lights
            num_active = random.randint(1, 2)
            action_indices = random.sample(range(self.output_size), num_active)
            active_lights = [i in action_indices for i in range(self.output_size)]
            dummy_probs = np.array([0.5 if active else 0.0 for active in active_lights])
            return active_lights, dummy_probs

    def calculate_step_reward(
        self,
        avg_speed,
        vehicle_count,
        crashes_this_step,
        waiting_vehicles_this_step,
        emissions_this_step,
    ):
        """
        Calculate the reward for the current simulation step.
        Args:
            avg_speed (float): Average speed of all vehicles in this step.
            vehicle_count (int): Total number of vehicles in this step.
            crashes_this_step (int): Number of crashes that occurred in this step.
            waiting_vehicles_this_step (int): Number of vehicles waiting at red lights this step.
            emissions_this_step (float): Total emissions generated by vehicles this step.
        Returns:
            float: The calculated reward for this step.
        """
        # Reward components (weights can be tuned)
        speed_reward_weight = 0.5
        waiting_penalty_weight = -0.2
        emission_penalty_weight = -0.01
        crash_penalty = +100.0  # Large penalty per crash

        reward = 0.0

        # Reward for speed (higher is better, scaled)
        # Normalize speed based on a typical max (e.g., 10) or use raw value scaled
        reward += speed_reward_weight * avg_speed

        # Penalty for waiting vehicles (more waiting is bad)
        reward += waiting_penalty_weight * waiting_vehicles_this_step

        # Penalty for emissions
        reward += emission_penalty_weight * emissions_this_step

        # Large penalty for crashes
        reward += crash_penalty * crashes_this_step

        # Optional: Penalty for frequent light changes (requires tracking previous action)

        # Scaling/Clamping (optional but can help stability)
        # reward = np.clip(reward, -10, 10) # Example clamp

        return reward

    def remember_step(self, state, actual_action, reward, log_prob):
        """Store data from a single step for the current epoch."""
        self.epoch_states.append(state)
        self.epoch_actual_actions.append(actual_action)
        self.epoch_rewards.append(reward)
        self.epoch_log_probs.append(log_prob)  # Store log_prob directly

    @tf.function  # Compile for potential speedup
    def _compute_loss_and_grads(self, states, actual_actions, discounted_returns):
        """Computes loss and gradients using GradientTape."""
        with tf.GradientTape() as tape:
            # Forward pass to get current action probabilities for the states visited
            action_probs = self.model(states, training=True)

            # Calculate log probabilities of the actual actions taken
            # Binary Cross Entropy: -(y*log(p) + (1-y)*log(1-p))
            # We need log(p) if action=1, log(1-p) if action=0
            log_probs_all = tf.math.log(
                action_probs + 1e-10
            ) * actual_actions + tf.math.log(1 - action_probs + 1e-10) * (
                1 - actual_actions
            )
            # Sum log probs across the action dimensions (lights)
            log_probs_taken = tf.reduce_sum(log_probs_all, axis=1)

            # Calculate the REINFORCE loss: - G_t * log(pi(A_t|S_t))
            # We sum this over the batch (epoch)
            loss = -tf.reduce_mean(discounted_returns * log_probs_taken)

        # Compute gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def train_epoch(self):
        """
        Train the model at the end of an epoch using the REINFORCE algorithm.
        """
        if self.model is None or not self.epoch_states:
            if not PERFORMANCE_MODE:
                print(
                    "Skipping training: No model or no states/rewards recorded for the epoch."
                )
            return

        if not PERFORMANCE_MODE:
            print(f"Starting REINFORCE training for Epoch {self.current_epoch}...")

        # 1. Calculate Discounted Returns (G_t)
        discounted_returns = []
        cumulative_return = 0.0
        for reward in reversed(self.epoch_rewards):
            cumulative_return = reward + self.discount_factor * cumulative_return
            discounted_returns.insert(0, cumulative_return)  # Add to front

        # Convert lists to tensors
        states_tensor = tf.convert_to_tensor(self.epoch_states, dtype=tf.float32)
        actual_actions_tensor = tf.convert_to_tensor(
            self.epoch_actual_actions, dtype=tf.float32
        )
        discounted_returns_tensor = tf.convert_to_tensor(
            discounted_returns, dtype=tf.float32
        )

        # Optional: Normalize returns (can improve stability)
        returns_mean = tf.reduce_mean(discounted_returns_tensor)
        returns_std = tf.math.reduce_std(discounted_returns_tensor)
        normalized_returns = (discounted_returns_tensor - returns_mean) / (
            returns_std + 1e-8
        )  # Add epsilon for stability

        # 2. Compute Loss and Gradients using GradientTape
        try:
            loss, grads = self._compute_loss_and_grads(
                states_tensor, actual_actions_tensor, normalized_returns
            )

            # 3. Apply Gradients
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if not PERFORMANCE_MODE:
                print(
                    f"Epoch {self.current_epoch} training complete. Loss: {loss.numpy():.4f}"
                )

        except Exception as e:
            print(
                f"Error during REINFORCE training for epoch {self.current_epoch}: {e}"
            )
            # Consider logging the traceback for debugging
            # import traceback
            # traceback.print_exc()

        # 4. Clear memory for the next epoch (Moved to end_epoch)
        # self.epoch_states = []
        # self.epoch_actual_actions = []
        # self.epoch_rewards = []
        # self.epoch_log_probs = []

    def end_epoch(self):
        """Handle end of epoch logic, including training and metric reset."""
        if not PERFORMANCE_MODE:
            print(
                f"\n--- Reached end of Epoch {self.current_epoch}. Processing... ---",
                flush=True,
            )

        # --- Train the model based on the completed epoch --- (REINFORCE)
        self.train_epoch()
        # --- Training finished ---

        # Calculate average speed for the epoch summary
        avg_speed_epoch = self.sum_avg_speeds_epoch / max(1, self.vehicle_updates_epoch)

        # Format summary message
        if not PERFORMANCE_MODE:
            print("=" * 60)
            print(f"EPOCH {self.current_epoch}/{self.total_epochs} SUMMARY")
            print("-" * 60)
            # Use accumulated step rewards for total reward
            print(f"Total Reward:  {self.epoch_accumulated_reward:.2f}")
            print(f"Avg Speed:     {avg_speed_epoch:.2f} units/step")
            print(f"Crashes:       {self.total_crashes_epoch}")
            print(
                f"Waiting Steps: {self.total_waiting_steps_epoch}"
            )  # Sum of waiting vehicles each step
            print(f"CO2 Emissions: {self.total_emissions_epoch:.2f} units")
            # Visualization status
            if (
                self.current_epoch
            ) % self.render_interval == 0 and self.current_epoch > 0:
                print("-" * 60)
                print(
                    "VISUALIZATION ENABLED for next epoch - Press SPACE to continue training"
                )
            print("=" * 60 + "\n")
        else:
            # Minimal summary for performance mode
            epoch_num_str = f"Epoch {self.current_epoch}/{self.total_epochs}"
            reward_str = f"Reward: {self.epoch_accumulated_reward:<10.2f}"
            crashes_str = f"Crashes: {self.total_crashes_epoch:<4}"
            avg_speed_str = f"AvgSpeed: {avg_speed_epoch:<5.2f}"
            print(
                f"| {epoch_num_str:<15} | {reward_str} | {crashes_str} | {avg_speed_str} |",
                flush=True,
            )

        # --- Increment epoch counter --- (Do this *before* resetting metrics for next epoch)
        self.current_epoch += 1

        # --- Reset metrics & data stores for the *next* epoch ---
        self.epoch_accumulated_reward = 0
        self.epoch_steps = 0
        self.total_crashes_epoch = 0
        self.total_waiting_steps_epoch = 0
        self.total_emissions_epoch = 0
        self.sum_avg_speeds_epoch = 0
        self.vehicle_updates_epoch = 0
        # Clear REINFORCE data stores
        self.epoch_states.clear()
        self.epoch_actual_actions.clear()
        self.epoch_rewards.clear()
        self.epoch_log_probs.clear()

        # Determine rendering for the *next* epoch
        if self.current_epoch % self.render_interval == 0 and self.current_epoch > 0:
            self.show_render = True
            self.waiting_for_space = True
        elif (
            self.total_epochs == 1 and self.current_epoch == 0
        ):  # Render if only one epoch
            self.show_render = True
            self.waiting_for_space = True
        else:
            self.show_render = False  # Ensure render is off otherwise
            self.waiting_for_space = False

    def should_render(self):
        """Determine if the simulation should be rendered"""
        return self.show_render

    def check_for_space(self):
        """Check if space was pressed to continue training"""
        # Only check if we are actually waiting
        if not self.waiting_for_space:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Allow quitting during wait
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if not PERFORMANCE_MODE:
                    print("Resuming training...")
                self.waiting_for_space = False
                self.show_render = False
                # Consume any other pending events to avoid double triggers
                pygame.event.clear()
                return True
        return False

    def update(
        self,
        scan_zones,
        avg_speed,  # Avg speed THIS STEP
        crashes_this_step,  # Crashes THIS STEP (returned by vehicle.move)
        waiting_vehicles,  # Waiting vehicles THIS STEP
        emissions_this_step,  # Emissions THIS STEP (summed from vehicle.update_emission)
        vehicle_count,  # Vehicle count THIS STEP
    ):
        """
        Main update function called during simulation (REINFORCE version).
        Gets state, predicts action, calculates step reward, stores trajectory data.
        Returns:
        - list: active_lights (booleans)
        """
        try:
            # --- Aggregate metrics for epoch summary --- (Separate from step reward)
            self.total_crashes_epoch += crashes_this_step
            self.total_waiting_steps_epoch += waiting_vehicles
            self.total_emissions_epoch += emissions_this_step
            if vehicle_count > 0:
                self.sum_avg_speeds_epoch += avg_speed
                self.vehicle_updates_epoch += 1

            # --- RL Step ---
            # 1. Get State
            state = self.get_state(scan_zones)

            # 2. Get Action (Probabilities and Thresholded)
            if (
                not PERFORMANCE_MODE and self.total_steps % 1000 == 0
            ):  # Less frequent logging
                print(f"  Step {self.total_steps}: Getting action...", flush=True)
            active_lights, action_probs = self.get_action(state)
            # active_lights is the actual action taken, e.g. [True, False, True, False]

            # 3. Calculate Step Reward
            step_reward = self.calculate_step_reward(
                avg_speed,
                vehicle_count,
                crashes_this_step,
                waiting_vehicles,
                emissions_this_step,
            )
            self.epoch_accumulated_reward += step_reward  # Accumulate for epoch summary

            # 4. Calculate Log Probability of Action Taken
            # Needed for REINFORCE loss. Avoid recomputing later if possible.
            # log(p) if action=1, log(1-p) if action=0
            action_tensor = tf.convert_to_tensor([active_lights], dtype=tf.float32)
            probs_tensor = tf.convert_to_tensor([action_probs], dtype=tf.float32)
            log_probs_all_step = tf.math.log(
                probs_tensor + 1e-10
            ) * action_tensor + tf.math.log(1 - probs_tensor + 1e-10) * (
                1 - action_tensor
            )
            log_prob_step = tf.reduce_sum(log_probs_all_step, axis=1).numpy()[0]

            # 5. Store Trajectory Data
            self.remember_step(state, active_lights, step_reward, log_prob_step)

            # --- Update step counts ---
            self.total_steps += 1
            self.epoch_steps += 1

            # Return the list of active lights (booleans)
            return active_lights

        except Exception as e:
            print(f"ERROR in neural controller update: {e}")
            # Fallback: Randomly activate 1 or 2 lights
            num_active = random.randint(1, 2)
            action_indices = random.sample(range(self.output_size), num_active)
            active_lights = [i in action_indices for i in range(self.output_size)]
            return active_lights

    def save_model(self, filepath):
        """Saves the model weights to the specified filepath."""
        if self.model:
            try:
                self.model.save_weights(filepath)
                logger.info(f"Model weights saved to {filepath}")
            except Exception as e:
                logger.error(f"Error saving model weights to {filepath}: {e}")
        else:
            logger.warning("Attempted to save model, but model is None.")

    def load_model(self, filepath):
        """Loads model weights from the specified filepath."""
        if self.model:
            try:
                self.model.load_weights(filepath)
                logger.info(f"Model weights loaded from {filepath}")
            except Exception as e:
                logger.error(
                    f"Error loading model weights from {filepath}: {e}. Ensure the model architecture matches."
                )
                # Optionally re-raise or handle differently
        else:
            logger.error(
                "Attempted to load weights, but model is None. Build the model first."
            )


# Helper function to extract vehicles from zones in the simulation
def get_vehicles_in_zones(direction_numbers, vehicles, DEFAULT_SCAN_ZONE_CONFIG):
    """
    Get all vehicles in each scan zone

    Returns:
    - List of 4 lists containing vehicle data for each zone
    """
    zone_directions = ["right", "down", "left", "up"]
    all_zone_vehicles = []

    # Define fallback dimensions and distance calculation logic using dictionaries
    fallback_dimensions = {
        "car": (40, 40),
        "bus": (60, 60),
        "truck": (60, 60),
        "bike": (20, 20),
    }
    default_dims = (40, 40)  # Default if type not found

    distance_calculators = {
        "right": lambda v, cam: v.x - cam["x"],
        "left": lambda v, cam: cam["x"]
        - (
            v.x + v.image.get_rect().width
            if hasattr(v, "image") and v.image
            else v.x + fallback_dimensions.get(v.vehicleClass, default_dims)[0]
        ),
        "down": lambda v, cam: cam["y"]
        - (
            v.y + v.image.get_rect().height
            if hasattr(v, "image") and v.image
            else v.y + fallback_dimensions.get(v.vehicleClass, default_dims)[1]
        ),
        "up": lambda v, cam: v.y - cam["y"],
    }

    for direction in zone_directions:
        scan_zone = DEFAULT_SCAN_ZONE_CONFIG[direction]
        camera = scan_zone["camera"]
        zone = scan_zone["zone"]
        vehicles_in_zone = []
        calculate_distance = distance_calculators[direction]  # Get the correct lambda

        for d in direction_numbers.values():
            for lane in range(3):
                for vehicle in vehicles[d][lane]:
                    try:
                        vehicle_left = vehicle.x
                        vehicle_top = vehicle.y

                        # Get dimensions: Use image rect if available, else use fallback dict
                        if hasattr(vehicle, "image") and vehicle.image is not None:
                            vehicle_width = vehicle.image.get_rect().width
                            vehicle_height = vehicle.image.get_rect().height
                        else:
                            vehicle_width, vehicle_height = fallback_dimensions.get(
                                vehicle.vehicleClass, default_dims
                            )

                        vehicle_right = vehicle_left + vehicle_width
                        vehicle_bottom = vehicle_top + vehicle_height

                        # Check if any part of the vehicle is in this scan zone
                        in_zone = not (
                            vehicle_right < zone["x1"]
                            or vehicle_left > zone["x2"]
                            or vehicle_bottom < zone["y1"]
                            or vehicle_top > zone["y2"]
                        )

                        # Calculate distance using the appropriate function from the dictionary
                        distance = calculate_distance(vehicle, camera)

                        # Include all vehicles in the zone
                        if in_zone:
                            vehicles_in_zone.append(
                                {
                                    "vehicle": vehicle,
                                    "distance": abs(distance),
                                    "speed": vehicle.speed,
                                    "type": vehicle.vehicleClass,
                                    "acceleration": (
                                        1
                                        if vehicle.accelerated
                                        else (-1 if vehicle.decelerated else 0)
                                    ),
                                    "position": (vehicle.x, vehicle.y),
                                }
                            )
                    except Exception as e:
                        # Skip this vehicle if any error occurs without printing
                        # Consider logging the error here if debugging is needed:
                        # logger.warning(f"Skipping vehicle due to error in zone calculation: {e}")
                        continue

        all_zone_vehicles.append(vehicles_in_zone)

    return all_zone_vehicles
