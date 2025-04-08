import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import random
from collections import deque


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
        self.render_interval = 10  # epochs between renders

        # State tracking for epoch-based training
        self.epoch_states = []
        self.epoch_actions = []  # Store the raw probabilities output by the model

        # Standard state tracking
        self.total_steps = 0
        self.current_epoch = 0
        self.epoch_reward = 0
        self.epoch_steps = 0
        self.show_render = False
        self.waiting_for_space = False

        # Performance metrics
        self.total_crashes_epoch = 0  # Track crashes per epoch for reward calculation
        self.total_waiting_epoch = 0  # Track waiting per epoch
        self.total_emissions_epoch = 0  # Track emissions per epoch
        self.total_speed_epoch = 0  # Track speed per epoch
        self.vehicle_count_epoch = 0  # Track vehicle count per epoch

        # Initialize the model with error handling
        try:
            print("Initializing neural network model...")
            self.model = self._build_model()
            print("Neural network model initialized successfully")
        except Exception as e:
            print(f"ERROR initializing neural network: {e}")
            print("Using random action fallback instead of neural network")
            self.model = None

    def _build_model(self):
        """Build a simple neural network model"""
        try:
            model = Sequential(
                [
                    Flatten(input_shape=self.input_shape),
                    Dense(256, activation="relu"),
                    Dense(128, activation="relu"),
                    # Sigmoid activation for multi-label probability (0 to 1)
                    Dense(self.output_size, activation="sigmoid"),
                ]
            )
            # Using binary_crossentropy as it's suitable for multi-label probabilities
            model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=self.learning_rate),
            )
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
        - tuple: (list of booleans for active lights, raw probabilities)
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
            print(f"    Step {self.total_steps}: Calling model.predict...", flush=True)
            action_probs = self.model.predict(np.array([state]), verbose=0)[0]
            print(
                f"    Step {self.total_steps}: Got prediction: {action_probs}",
                flush=True,
            )

            # Apply threshold to determine active lights
            active_lights = (
                action_probs >= 0.5
            ).tolist()  # Convert numpy bool array to list

            # Ensure at least one light is green if all are below threshold?
            # Or allow all red? Let's allow all red for now. If needed, can force one.
            # if not any(active_lights):
            #     active_lights[np.argmax(action_probs)] = True # Activate the highest probability one

            return active_lights, action_probs

        except Exception as e:
            print(f"Error getting action: {e}")
            # Fallback: Randomly activate 1 or 2 lights
            num_active = random.randint(1, 2)
            action_indices = random.sample(range(self.output_size), num_active)
            active_lights = [i in action_indices for i in range(self.output_size)]
            dummy_probs = np.array([0.5 if active else 0.0 for active in active_lights])
            return active_lights, dummy_probs

    def calculate_epoch_reward(self):
        """
        Calculate the total reward for the completed epoch based on aggregated metrics.
        (This replaces the step-by-step reward calculation)
        """
        avg_speed_epoch = self.total_speed_epoch / max(1, self.vehicle_count_epoch)

        # Use the same reward logic as before, but on epoch totals
        speed_reward = avg_speed_epoch * 5
        speed_reward = min(speed_reward, 1000)

        crash_penalty = -500 * min(
            self.total_crashes_epoch, 5 * self.steps_per_epoch
        )  # Scale max crashes allowed
        waiting_penalty = -0.5 * min(
            self.total_waiting_epoch, 1000 * self.steps_per_epoch
        )
        emission_penalty = -0.01 * min(
            self.total_emissions_epoch, 10000 * self.steps_per_epoch
        )

        total_reward = speed_reward + crash_penalty + waiting_penalty + emission_penalty

        # Clamp reward
        total_reward = max(
            min(total_reward, 5000 * self.steps_per_epoch), -5000 * self.steps_per_epoch
        )

        # Normalize reward (e.g., to [-1, 1] or [0, 1]) - simple scaling for now
        # This helps stabilize training when rewards vary wildly.
        # Max possible positive reward is roughly speed_reward (~1000)
        # Max possible negative reward is roughly crash_penalty + waiting + emission
        # Let's scale based on a reasonable range, e.g., -5000 to +1000 per 1000 steps
        max_possible_positive = 1000 * (self.steps_per_epoch / 1000)
        max_possible_negative = -5000 * (self.steps_per_epoch / 1000)

        if total_reward > 0:
            normalized_reward = total_reward / max_possible_positive
        elif total_reward < 0:
            # Map negative reward range to [-1, 0]
            normalized_reward = total_reward / abs(max_possible_negative)
        else:
            normalized_reward = 0

        # Clamp normalized reward to [-1, 1] just in case
        normalized_reward = max(min(normalized_reward, 1.0), -1.0)

        self.epoch_reward = total_reward  # Store the raw total reward for reporting

        return normalized_reward  # Return normalized reward for training

    def remember_step(self, state, action_probs):
        """Store state and action probabilities for the current epoch."""
        self.epoch_states.append(state)
        self.epoch_actions.append(action_probs)

    def train_epoch(self):
        """
        Train the model at the end of an epoch based on the overall epoch reward.
        This is a simplified approach, not standard RL, aiming to adjust weights based on epoch performance.
        """
        if self.model is None or not self.epoch_states:
            print("Skipping training: No model or no states recorded for the epoch.")
            return

        print(f"Starting training for Epoch {self.current_epoch}...")

        # Calculate the normalized reward for the entire epoch
        normalized_epoch_reward = self.calculate_epoch_reward()
        print(
            f"Epoch {self.current_epoch} Raw Reward: {self.epoch_reward:.2f}, Normalized Reward for Training: {normalized_epoch_reward:.4f}"
        )

        # Prepare training data
        states = np.array(self.epoch_states)
        recorded_action_probs = np.array(self.epoch_actions)

        # --- Simple Training Logic ---
        # Create target probabilities.
        # If epoch reward is positive, slightly increase the probability of actions taken.
        # If epoch reward is negative, slightly decrease the probability.
        # The magnitude of change is scaled by the normalized reward.

        # Target = CurrentProbs + LearningRate * NormalizedReward * (Gradient-like term)
        # Simplified: Target = CurrentProbs + LearningRate * NormalizedReward
        # Let's try pushing probabilities towards 1 if good reward, towards 0 if bad.

        target_actions = np.copy(recorded_action_probs)

        # Adjust target probabilities based on the overall epoch reward
        # If reward is positive, push probabilities slightly towards 1
        # If reward is negative, push probabilities slightly towards 0
        # Adjustment factor based on reward
        adjustment = self.learning_rate * normalized_epoch_reward

        # Apply adjustment: target = current + adjustment
        # We need to be careful not to just add the reward everywhere.
        # Let's adjust based on whether the prob was > 0.5 or < 0.5

        # Alternative: Simple supervised target based on reward
        if normalized_epoch_reward > 0.1:  # Treat as "good" epoch
            # Encourage the probabilities that were high?
            # Set target to 1 for actions with prob > 0.5, 0 otherwise? Too harsh.
            # Simple approach: Nudge all probabilities slightly towards extremes based on reward
            target_actions += adjustment  # Nudge probabilities up
        elif normalized_epoch_reward < -0.1:  # Treat as "bad" epoch
            target_actions += (
                adjustment  # Nudge probabilities down (since adjustment is negative)
            )

        # Clamp target probabilities to [0, 1]
        target_actions = np.clip(target_actions, 0.0, 1.0)

        try:
            print(
                f"Training model on {len(states)} samples from epoch {self.current_epoch}..."
            )
            history = self.model.fit(
                states, target_actions, epochs=1, verbose=0, batch_size=self.batch_size
            )
            loss = history.history["loss"][0]
            print(f"Epoch {self.current_epoch} training complete. Loss: {loss:.4f}")
        except Exception as e:
            print(f"Error during model training for epoch {self.current_epoch}: {e}")

        # Clear memory for the next epoch
        self.epoch_states = []
        self.epoch_actions = []

    def end_epoch(self):
        """Handle end of epoch logic, including training."""
        print(
            f"\n--- Reached end of Epoch {self.current_epoch}. Processing... ---",
            flush=True,  # FORCE FLUSH
        )  # Add entry print

        # --- Train the model based on the completed epoch ---
        self.train_epoch()
        # --- Training finished ---

        self.current_epoch += (
            1  # Increment epoch counter *after* training for the completed one
        )

        # Format a cleaner, more structured progress message using stored epoch reward
        print("=" * 60)
        print(
            f"EPOCH {self.current_epoch-1}/{self.total_epochs} SUMMARY"
        )  # Report for the epoch just finished
        print("-" * 60)
        print(f"Total Reward:  {self.epoch_reward:.2f}")  # Use the stored raw reward
        avg_speed = self.total_speed_epoch / max(1, self.vehicle_count_epoch)
        print(f"Avg Speed:     {avg_speed:.2f} units/step")
        print(f"Crashes:       {self.total_crashes_epoch}")
        print(
            f"Waiting Steps: {self.total_waiting_epoch}"
        )  # Sum of waiting vehicles each step
        print(f"CO2 Emissions: {self.total_emissions_epoch:.2f} units")
        # Epsilon removed

        # Add visualization status
        if (self.current_epoch - 1) % self.render_interval == 0 and (
            self.current_epoch - 1
        ) > 0:  # Check for previous epoch
            print("-" * 60)
            print(
                "VISUALIZATION ENABLED for next epoch - Press SPACE to continue training"
            )
        print("=" * 60 + "\n")

        # --- Reset metrics for the *next* epoch ---
        self.epoch_reward = 0  # Raw reward accumulator for next epoch reporting
        self.epoch_steps = 0
        # Reset epoch-specific performance metrics
        self.total_crashes_epoch = 0
        self.total_waiting_epoch = 0
        self.total_emissions_epoch = 0
        self.total_speed_epoch = 0
        self.vehicle_count_epoch = 0

        # Determine if we should render for the *next* epoch
        # Render every N epochs, but maybe not the very first one unless total_epochs=1
        if self.current_epoch % self.render_interval == 0 and self.total_epochs > 1:
            self.show_render = True
            self.waiting_for_space = True
        elif self.total_epochs == 1:  # Render if only one epoch
            self.show_render = True
            self.waiting_for_space = True

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
                print("Resuming training...")
                self.waiting_for_space = False
                self.show_render = False
                # Consume any other pending events to avoid double triggers
                pygame.event.clear()
                return True
        return False

    def update(
        self, scan_zones, avg_speed, crashes, waiting_vehicles, emissions, vehicle_count
    ):
        """
        Main update function called during simulation.
        Gets state, predicts action probabilities, determines active lights,
        and stores data for end-of-epoch training.
        """
        try:
            # --- Aggregate metrics for epoch reward calculation ---
            self.vehicle_count_epoch = max(self.vehicle_count_epoch, vehicle_count)
            self.total_speed_epoch += avg_speed  # Store sum of avg speeds
            self.total_crashes_epoch += crashes  # Accumulate crashes over the epoch
            self.total_waiting_epoch += (
                waiting_vehicles  # Accumulate waiting vehicles over epoch
            )
            self.total_emissions_epoch += emissions  # Accumulate emissions over epoch

            # Convert scan zone data to state
            state = self.get_state(scan_zones)

            # Get action probabilities and thresholded actions
            print(f"  Step {self.total_steps}: Calling get_action...", flush=True)
            active_lights, action_probs = self.get_action(state)
            print(f"  Step {self.total_steps}: Got action: {active_lights}", flush=True)

            # Store state and action probabilities for end-of-epoch training
            self.remember_step(state, action_probs)

            # --- Re-introduce infrequent progress print ---
            if self.total_steps % 5000 == 0 and self.total_steps > 0:
                epoch_progress = (self.epoch_steps / self.steps_per_epoch) * 100
                print(
                    f"---> Step {self.total_steps} | Epoch {self.current_epoch}/{self.total_epochs} ({epoch_progress:.1f}% complete)",
                    flush=True,  # FORCE FLUSH
                )

            self.total_steps += 1
            self.epoch_steps += 1

            # Return the list of active lights (booleans)
            return active_lights

        except Exception as e:
            # Log the error
            print(f"ERROR in neural controller update: {e}")
            # Fallback: Randomly activate 1 or 2 lights
            num_active = random.randint(1, 2)
            action_indices = random.sample(range(self.output_size), num_active)
            active_lights = [i in action_indices for i in range(self.output_size)]
            return active_lights


# Helper function to extract vehicles from zones in the simulation
def get_vehicles_in_zones(direction_numbers, vehicles, DEFAULT_SCAN_ZONE_CONFIG):
    """
    Get all vehicles in each scan zone

    Returns:
    - List of 4 lists containing vehicle data for each zone
    """
    # Debug info has been removed to prevent console spam

    zone_directions = ["right", "down", "left", "up"]
    all_zone_vehicles = []

    for direction in zone_directions:
        # Get scan zone configuration
        scan_zone = DEFAULT_SCAN_ZONE_CONFIG[direction]
        camera = scan_zone["camera"]
        zone = scan_zone["zone"]

        # Find all vehicles in the scan zone
        vehicles_in_zone = []

        for d in direction_numbers.values():
            for lane in range(3):
                for vehicle in vehicles[d][lane]:
                    try:
                        # Get vehicle coordinates
                        vehicle_left = vehicle.x
                        vehicle_top = vehicle.y

                        # Get dimensions using image.get_rect() as in the Vehicle class
                        if hasattr(vehicle, "image") and vehicle.image is not None:
                            vehicle_width = vehicle.image.get_rect().width
                            vehicle_height = vehicle.image.get_rect().height
                        else:
                            # Fallback dimensions based on vehicle type
                            if vehicle.vehicleClass == "car":
                                vehicle_width = vehicle_height = 40
                            elif (
                                vehicle.vehicleClass == "bus"
                                or vehicle.vehicleClass == "truck"
                            ):
                                vehicle_width = vehicle_height = 60
                            else:  # bike
                                vehicle_width = vehicle_height = 20

                        vehicle_right = vehicle_left + vehicle_width
                        vehicle_bottom = vehicle_top + vehicle_height

                        # Check if any part of the vehicle is in this scan zone
                        in_zone = not (
                            vehicle_right < zone["x1"]
                            or vehicle_left > zone["x2"]
                            or vehicle_bottom < zone["y1"]
                            or vehicle_top > zone["y2"]
                        )

                        # Calculate distance to camera
                        if direction == "right":
                            distance = vehicle_left - camera["x"]
                        elif direction == "left":
                            distance = camera["x"] - vehicle_right
                        elif direction == "down":
                            distance = camera["y"] - vehicle_bottom
                        else:  # up
                            distance = vehicle_top - camera["y"]

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
                        continue

        all_zone_vehicles.append(vehicles_in_zone)

    return all_zone_vehicles
