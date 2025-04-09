import numpy as np
import pygame
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.utils import PERFORMANCE_MODE
import logging

# --- PPO Hyperparameters --- #
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
CLIP_RATIO = 0.2  # PPO clipping ratio
ACTOR_LR = 3e-4  # Actor learning rate
CRITIC_LR = 1e-3  # Critic learning rate
TRAIN_EPOCHS_PER_UPDATE = 10  # Number of optimization epochs per learning update
TARGET_KL = 0.01  # Target KL divergence for early stopping
REPLAY_BUFFER_SIZE = 20000  # Max size of the replay buffer
MIN_BUFFER_SIZE = 1000  # Minimum transitions needed before learning starts
UPDATE_EVERY_N_STEPS = 4000  # How often to run the learning update
BATCH_SIZE = 64  # Batch size for sampling from buffer
# --- End PPO Hyperparameters --- #


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


class NeuralTrafficControllerPPO:  # Renamed class
    def __init__(self, steps_per_epoch=10000, total_epochs=50):
        # Configuration (Some original config might be less relevant for PPO step-based updates)
        self.steps_per_epoch = steps_per_epoch  # Still useful for reporting/saving
        self.total_epochs = total_epochs  # Total simulation duration
        self.input_shape = (4, 20, 5)
        self.output_size = 4
        self.render_interval = 10  # epochs between renders

        # PPO Specific State
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.actor_optimizer = Adam(learning_rate=ACTOR_LR)
        self.critic_optimizer = Adam(learning_rate=CRITIC_LR)

        # Standard state tracking
        self.total_steps = 0
        self.current_epoch = 0
        self.epoch_accumulated_reward = 0  # Still useful for epoch summary
        self.epoch_steps = 0
        self.show_render = False
        self.waiting_for_space = False

        # Aggregated metrics for end-of-epoch summary (reset each epoch)
        self.total_crashes_epoch = 0
        self.total_waiting_steps_epoch = 0
        self.total_emissions_epoch = 0
        self.sum_avg_speeds_epoch = 0
        self.vehicle_updates_epoch = 0

        # Initialize Actor-Critic models
        try:
            if not PERFORMANCE_MODE:
                print("Initializing PPO Actor-Critic models...")
            self.actor, self.critic = self._build_actor_critic_models()
            if not PERFORMANCE_MODE:
                print("PPO Actor-Critic models initialized successfully")
                # self.actor.summary()
                # self.critic.summary()
        except Exception as e:
            print(f"ERROR initializing PPO models: {e}")
            print("Using random action fallback instead.")
            self.actor = None
            self.critic = None

        # Store last state for TD calculations
        self.last_state = None
        self.last_action_log_prob = None
        self.last_value = None
        self.last_action = None

    def _build_actor_critic_models(self):
        """Builds separate Actor (policy) and Critic (value) networks."""
        # Input Layer
        input_layer = Input(shape=self.input_shape)
        flattened = Flatten()(input_layer)

        # Shared layers (optional, can be separate)
        shared_dense1 = Dense(256, activation="relu")(flattened)
        shared_dense2 = Dense(128, activation="relu")(shared_dense1)

        # --- Actor Head ---
        # Outputs logits for each action dimension (traffic light)
        # We use logits for numerical stability in loss calculation
        action_logits = Dense(self.output_size, activation=None, name="action_logits")(
            shared_dense2
        )
        actor = Model(inputs=input_layer, outputs=action_logits)

        # --- Critic Head ---
        # Outputs a single value representing the estimated state value
        state_value = Dense(1, activation=None, name="state_value")(shared_dense2)
        critic = Model(inputs=input_layer, outputs=state_value)

        return actor, critic

    def get_state(self, scan_zones):
        """Convert scan zone data to a state representation.

        Args:
            scan_zones: List of 4 lists containing vehicle data from each zone.

        Returns:
            State matrix of shape (4, 20, 5).
        """
        state = np.zeros(self.input_shape)
        for zone_idx, zone_vehicles in enumerate(scan_zones):
            zone_vehicles.sort(key=lambda x: x["distance"])
            for vehicle_idx, vehicle_data in enumerate(zone_vehicles[:20]):
                if vehicle_idx >= 20:
                    break
                vehicle = vehicle_data["vehicle"]
                vehicle_type_map = {"car": 0, "bus": 0.33, "truck": 0.67, "bike": 1}
                state[zone_idx, vehicle_idx, 0] = min(
                    vehicle_data["distance"] / 500, 1.0
                )
                state[zone_idx, vehicle_idx, 1] = min(vehicle_data["speed"] / 10, 1.0)
                state[zone_idx, vehicle_idx, 2] = vehicle_type_map.get(
                    vehicle_data["type"], 0
                )
                state[zone_idx, vehicle_idx, 3] = (vehicle_data["acceleration"] + 1) / 2
                state[zone_idx, vehicle_idx, 4] = min(vehicle.waiting_time / 100, 1.0)
        return state

    # @tf.function # Can compile for faster inference
    def get_action_and_value(self, state):
        """Get action, log probability, and value estimate for a given state."""
        if self.actor is None or self.critic is None:
            num_active = random.randint(1, 2)
            action_indices = random.sample(range(self.output_size), num_active)
            action = [
                1.0 if i in action_indices else 0.0 for i in range(self.output_size)
            ]
            return np.array(action), 0.0, 0.0

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_logits = self.actor(state_tensor, training=False)
        value = self.critic(state_tensor, training=False)[0, 0]
        action_probs = tf.sigmoid(action_logits)[0]
        action = tf.cast(
            tf.random.uniform(shape=(self.output_size,)) < action_probs,
            dtype=tf.float32,
        )
        probs_tensor = tf.clip_by_value(action_probs, 1e-10, 1.0 - 1e-10)
        log_probs_all = action * tf.math.log(probs_tensor) + (
            1.0 - action
        ) * tf.math.log(1.0 - probs_tensor)
        log_prob = tf.reduce_sum(log_probs_all)

        return action.numpy(), log_prob.numpy(), value.numpy()

    def calculate_step_reward(
        self,
        avg_speed,
        vehicle_count,
        crashes_this_step,
        waiting_vehicles_this_step,
        emissions_this_step,
    ):
        """Calculate the raw reward for the current simulation step."""
        speed_reward_weight = 0.5
        waiting_penalty_weight = -0.2
        emission_penalty_weight = -0.01
        crash_penalty = -100.0  # Negative penalty for crashes

        reward = 0.0
        reward += speed_reward_weight * avg_speed
        reward += waiting_penalty_weight * waiting_vehicles_this_step
        reward += emission_penalty_weight * emissions_this_step
        reward += crash_penalty * crashes_this_step
        return reward

    def store_transition(
        self, state, action, reward, next_state, done, log_prob, value
    ):
        """Stores a step transition in the replay buffer."""
        # Add terminal state handling if needed (e.g., if next_state is terminal)
        self.replay_buffer.append(
            (state, action, reward, next_state, done, log_prob, value)
        )

    # @tf.function # Can compile for faster training step
    def _compute_ppo_losses(self, states, actions, log_probs_old, returns, advantages):
        """Computes Actor (policy) and Critic (value) losses for a batch."""
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            values_pred = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - values_pred))
            action_logits = self.actor(states, training=True)
            action_probs = tf.sigmoid(action_logits)
            action_probs_clipped = tf.clip_by_value(action_probs, 1e-10, 1.0 - 1e-10)
            log_probs_current_all = actions * tf.math.log(action_probs_clipped) + (
                1.0 - actions
            ) * tf.math.log(1.0 - action_probs_clipped)
            log_probs_current = tf.reduce_sum(log_probs_current_all, axis=1)
            ratios = tf.exp(log_probs_current - log_probs_old)
            clipped_ratios = tf.clip_by_value(
                ratios, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO
            )
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(
            critic_loss, self.critic.trainable_variables
        )
        approx_kl = tf.reduce_mean(log_probs_old - log_probs_current)

        return actor_loss, critic_loss, actor_grads, critic_grads, approx_kl

    def learn_ppo(self):
        """Perform PPO learning update using data from the replay buffer."""
        if (
            len(self.replay_buffer) < BATCH_SIZE
            or len(self.replay_buffer) < MIN_BUFFER_SIZE
        ):
            # print("Buffer too small, skipping learning.")
            return  # Not enough data yet

        if not PERFORMANCE_MODE:
            print(
                f"Starting PPO Learning Update (Buffer size: {len(self.replay_buffer)})...",
                flush=True,
            )

        # --- Sample a batch from the buffer --- #
        # For simplicity, sample the whole buffer if it's small, or a random batch
        batch_indices = np.random.choice(
            len(self.replay_buffer),
            size=min(BATCH_SIZE, len(self.replay_buffer)),
            replace=False,
        )
        batch = [self.replay_buffer[i] for i in batch_indices]
        # Alternatively, use the whole buffer for on-policy feel if UPDATE_EVERY_N_STEPS is tuned
        # batch = list(self.replay_buffer)
        # self.replay_buffer.clear() # Clear buffer after use for on-policy feel

        states, actions, rewards, next_states, dones, log_probs_old, values_old = map(
            np.array, zip(*batch)
        )

        # Convert actions to float32 if they aren't already
        actions = actions.astype(np.float32)
        rewards = rewards.astype(np.float32)
        dones = dones.astype(np.float32)
        log_probs_old = log_probs_old.astype(np.float32)
        values_old = values_old.astype(np.float32)
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)

        # --- Calculate GAE (Generalized Advantage Estimation) and Returns --- #
        next_values = self.critic(next_states, training=False)[:, 0].numpy()
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        last_value = next_values[-1]  # Bootstrap from the last next_state

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * last_value * mask - values_old[t]
            advantages[t] = last_advantage = (
                delta + GAMMA * GAE_LAMBDA * mask * last_advantage
            )
            last_value = values_old[t]

        # Calculate returns (targets for value function)
        returns = advantages + values_old

        # Normalize advantages (optional but recommended)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # --- Convert to Tensors --- #
        states_tensor = tf.convert_to_tensor(states)
        actions_tensor = tf.convert_to_tensor(actions)
        log_probs_old_tensor = tf.convert_to_tensor(log_probs_old)
        returns_tensor = tf.convert_to_tensor(returns)
        advantages_tensor = tf.convert_to_tensor(advantages)

        # --- PPO Optimization Loop --- #
        for i in range(TRAIN_EPOCHS_PER_UPDATE):
            actor_loss, critic_loss, actor_grads, critic_grads, approx_kl = (
                self._compute_ppo_losses(
                    states_tensor,
                    actions_tensor,
                    log_probs_old_tensor,
                    returns_tensor,
                    advantages_tensor,
                )
            )

            # Apply gradients
            self.actor_optimizer.apply_gradients(
                zip(actor_grads, self.actor.trainable_variables)
            )
            self.critic_optimizer.apply_gradients(
                zip(critic_grads, self.critic.trainable_variables)
            )

            # Early stopping based on KL divergence
            if approx_kl > 1.5 * TARGET_KL:
                if not PERFORMANCE_MODE:
                    print(
                        f"  Early stopping at epoch {i+1}/{TRAIN_EPOCHS_PER_UPDATE} due to KL divergence ({approx_kl:.4f})",
                        flush=True,
                    )
                break

        if not PERFORMANCE_MODE:
            print(
                f"  PPO Update finished. Actor Loss: {actor_loss.numpy():.4f}, Critic Loss: {critic_loss.numpy():.4f}, Approx KL: {approx_kl.numpy():.4f}",
                flush=True,
            )

        # Clear buffer if using on-policy style update
        if BATCH_SIZE >= len(
            self.replay_buffer
        ):  # Heuristic: if we likely used the whole buffer
            self.replay_buffer.clear()

    def end_epoch(self):  # Kept for reporting and rendering logic
        """Handle end of epoch logic, primarily for reporting and triggering renders."""
        if not PERFORMANCE_MODE:
            print(
                f"\n--- Reached end of Epoch {self.current_epoch}. Processing... ---",
                flush=True,
            )

        # --- NO TRAINING HERE IN PPO (handled by learn_ppo periodically) --- #

        avg_speed_epoch = self.sum_avg_speeds_epoch / max(1, self.vehicle_updates_epoch)

        # Format summary message
        if not PERFORMANCE_MODE:
            print("=" * 60)
            print(f"EPOCH {self.current_epoch}/{self.total_epochs} SUMMARY")
            print("-" * 60)
            print(f"Total Reward:  {self.epoch_accumulated_reward:.2f}")
            print(f"Avg Speed:     {avg_speed_epoch:.2f} units/step")
            print(f"Crashes:       {self.total_crashes_epoch}")
            print(f"Waiting Steps: {self.total_waiting_steps_epoch}")
            print(f"CO2 Emissions: {self.total_emissions_epoch:.2f} units")
            if (
                self.current_epoch + 1
            ) % self.render_interval == 0 and self.current_epoch >= 0:
                print("-" * 60)
                print(
                    "VISUALIZATION WILL BE ENABLED for next epoch - Press SPACE to continue training"
                )
            print("=" * 60 + "\n")
        else:
            epoch_num_str = f"Epoch {self.current_epoch}/{self.total_epochs}"
            reward_str = f"Reward: {self.epoch_accumulated_reward:<10.2f}"
            crashes_str = f"Crashes: {self.total_crashes_epoch:<4}"
            avg_speed_str = f"AvgSpeed: {avg_speed_epoch:<5.2f}"
            print(
                f"| {epoch_num_str:<15} | {reward_str} | {crashes_str} | {avg_speed_str} |",
                flush=True,
            )

        self.current_epoch += 1

        # Reset metrics for the *next* epoch summary
        self.epoch_accumulated_reward = 0
        self.epoch_steps = 0
        self.total_crashes_epoch = 0
        self.total_waiting_steps_epoch = 0
        self.total_emissions_epoch = 0
        self.sum_avg_speeds_epoch = 0
        self.vehicle_updates_epoch = 0

        # Determine rendering for the *next* epoch
        if self.current_epoch % self.render_interval == 0 and self.current_epoch > 0:
            self.show_render = True
            self.waiting_for_space = True
        elif self.total_epochs == 1 and self.current_epoch == 1:
            self.show_render = True  # Render if only one epoch total (adjust check)
            self.waiting_for_space = True
        else:
            self.show_render = False
            self.waiting_for_space = False

    def should_render(self):
        """Determine if the simulation should be rendered."""
        return self.show_render

    # @cython.compile / @numba.jit (If Pygame interaction removed/conditional)
    def check_for_space(self):
        """Check if space was pressed to continue training."""
        # Only check if we are actually waiting AND rendering
        if not self.waiting_for_space or not self.show_render:
            return False

        # Only interact with Pygame events if rendering is shown
        if self.show_render and pygame.display.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Allow quitting during wait
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if not PERFORMANCE_MODE:
                        print("Resuming training...")
                    self.waiting_for_space = False
                    self.show_render = False
                    pygame.event.clear()
                    return True
        return False

    def update(  # Renamed from PPO perspective (this is effectively 'step')
        self,
        scan_zones,  # Current scan zone info
        avg_speed,  # Avg speed THIS STEP (before move)
        crashes_this_step,  # Crashes THIS STEP (during move)
        waiting_vehicles,  # Waiting vehicles THIS STEP (before move)
        emissions_this_step,  # Emissions THIS STEP (before move)
        vehicle_count,  # Vehicle count THIS STEP (before move)
        # Add done flag if applicable from environment
    ):
        """PPO Step: Get action, store transition, trigger learning.

        Returns:
            Action taken (list of booleans).
        """
        # --- Aggregate metrics for epoch summary (before state calculation) --- #
        self.total_crashes_epoch += crashes_this_step
        self.total_waiting_steps_epoch += waiting_vehicles
        self.total_emissions_epoch += emissions_this_step
        if vehicle_count > 0:
            self.sum_avg_speeds_epoch += avg_speed
            self.vehicle_updates_epoch += 1

        # --- PPO Step --- #
        # 1. Get State representation from scan zones
        current_state = self.get_state(scan_zones)

        # 2. Get Action, Log Prob, and Value from Actor-Critic
        action_floats, log_prob, value = self.get_action_and_value(current_state)
        # Convert action floats (0.0/1.0) to booleans for simulation use
        action_taken_bools = (action_floats > 0.5).tolist()

        # 3. Store the *previous* step's transition (if available)
        # We need current_state as the 'next_state' for the previous transition
        if self.last_state is not None:
            # Calculate reward for the transition that *led* to current_state
            step_reward = self.calculate_step_reward(
                avg_speed,
                vehicle_count,
                crashes_this_step,
                waiting_vehicles,
                emissions_this_step,
            )
            self.epoch_accumulated_reward += step_reward  # Accumulate for epoch summary

            # Assume 'done' is False for now unless environment provides it
            done = False
            self.store_transition(
                self.last_state,
                self.last_action,
                step_reward,
                current_state,
                done,
                self.last_action_log_prob,
                self.last_value,
            )

        # 4. Update last known values for the *next* transition
        self.last_state = current_state
        self.last_action = action_floats  # Store float action used for log_prob calc
        self.last_action_log_prob = log_prob
        self.last_value = value

        # 5. Trigger PPO learning update periodically
        self.total_steps += 1
        self.epoch_steps += 1
        if self.total_steps % UPDATE_EVERY_N_STEPS == 0:
            self.learn_ppo()

        # Return the boolean action for the simulation to execute
        return action_taken_bools

    def save_model(self, filepath_prefix):
        """Saves the actor and critic model weights."""
        if self.actor and self.critic:
            try:
                actor_path = f"{filepath_prefix}_actor.weights.h5"
                critic_path = f"{filepath_prefix}_critic.weights.h5"
                self.actor.save_weights(actor_path)
                self.critic.save_weights(critic_path)
                logger.info(f"PPO models saved to {filepath_prefix} prefix")
            except Exception as e:
                logger.error(
                    f"Error saving PPO models with prefix {filepath_prefix}: {e}"
                )
        else:
            logger.warning("Attempted to save PPO models, but one or both are None.")

    def load_model(self, filepath_prefix):
        """Loads actor and critic model weights."""
        if self.actor and self.critic:
            try:
                actor_path = f"{filepath_prefix}_actor.weights.h5"
                critic_path = f"{filepath_prefix}_critic.weights.h5"
                if os.path.exists(actor_path) and os.path.exists(critic_path):
                    self.actor.load_weights(actor_path)
                    self.critic.load_weights(critic_path)
                    logger.info(f"PPO models loaded from {filepath_prefix} prefix")
                else:
                    logger.error(
                        f"Cannot load PPO models: File(s) not found for prefix {filepath_prefix}"
                    )
            except Exception as e:
                logger.error(
                    f"Error loading PPO models from {filepath_prefix}: {e}. Ensure architecture matches."
                )
        else:
            logger.error(
                "Attempted to load PPO weights, but models are None. Build models first."
            )


# Helper function (potential optimization target)
# @cython.compile / @numba.jit
def get_vehicles_in_zones(
    direction_numbers: dict, vehicles: dict, DEFAULT_SCAN_ZONE_CONFIG: dict
) -> list:
    """Get all vehicles in each scan zone.

    Args:
        direction_numbers: Mapping from direction name to number.
        vehicles: Dictionary of vehicles by direction and lane.
        DEFAULT_SCAN_ZONE_CONFIG: Configuration for scan zones.

    Returns:
        List of 4 lists containing vehicle data for each zone.
    """
    zone_directions = ["right", "down", "left", "up"]
    all_zone_vehicles = []

    fallback_dimensions = {
        "car": (40, 40),
        "bus": (60, 60),
        "truck": (60, 60),
        "bike": (20, 20),
    }
    default_dims = (40, 40)

    # Corrected distance calculators (simpler version)
    distance_calculators = {
        "right": lambda v, cam, dims: v.x - cam["x"],
        "left": lambda v, cam, dims: cam["x"] - (v.x + dims[0]),
        "down": lambda v, cam, dims: cam["y"] - (v.y + dims[1]),
        "up": lambda v, cam, dims: v.y - cam["y"],
    }

    for direction in zone_directions:
        scan_zone = DEFAULT_SCAN_ZONE_CONFIG[direction]
        camera = scan_zone["camera"]
        zone = scan_zone["zone"]
        vehicles_in_zone = []
        calculate_distance_lambda = distance_calculators[direction]

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
                        current_dims = (vehicle_width, vehicle_height)

                        vehicle_right = vehicle_left + vehicle_width
                        vehicle_bottom = vehicle_top + vehicle_height

                        in_zone = not (
                            vehicle_right < zone["x1"]
                            or vehicle_left > zone["x2"]
                            or vehicle_bottom < zone["y1"]
                            or vehicle_top > zone["y2"]
                        )

                        # Calculate distance using the appropriate function and current dims
                        distance = calculate_distance_lambda(
                            vehicle, camera, current_dims
                        )

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
                        # logger.warning(f"Skipping vehicle due to error in zone calculation: {e}")
                        continue
        all_zone_vehicles.append(vehicles_in_zone)
    return all_zone_vehicles
