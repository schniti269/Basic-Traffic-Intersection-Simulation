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
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.utils import PERFORMANCE_MODE
import logging

# --- PPO Hyperparameters --- #
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
CLIP_RATIO = 0.15  # PPO clipping ratio - reduziert für höhere Stabilität
ACTOR_LR = 1e-4  # Actor learning rate (weiter reduziert)
CRITIC_LR = 5e-4  # Critic learning rate (weiter reduziert)
TRAIN_EPOCHS_PER_UPDATE = 8  # Number of optimization epochs (reduziert)
TARGET_KL = 0.015  # Target KL divergence - etwas erhöht für weniger konservatives Lernen
REPLAY_BUFFER_SIZE = 20000  # Max size of the replay buffer
MIN_BUFFER_SIZE = 2000  # Minimum transitions needed before learning starts (erhöht)
UPDATE_EVERY_N_STEPS = 1000  # How often to run the learning update (reduziert für häufigere Updates)
# --- End PPO Hyperparameters --- #

# --- New State Representation Config --- #
MAX_VEHICLES = 50  # Maximum number of vehicles to consider in the state
VEHICLE_FEATURES = 4  # distance, speed, wait_time, angle
FLAT_STATE_SIZE = MAX_VEHICLES * VEHICLE_FEATURES
INTERSECTION_CENTER_X = 700  # Approx center based on typical 1400 width
INTERSECTION_CENTER_Y = 400  # Approx center based on typical 800 height
# --- End New State Config --- #
logger = logging.getLogger(__name__)


class NeuralTrafficControllerPPO:  # Renamed class
    def __init__(self, steps_per_epoch=10000, total_epochs=50):
        # Configuration
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.input_size = FLAT_STATE_SIZE  # NEW FLAT INPUT SIZE
        self.output_size = 4  # Assuming 4 traffic light phases/actions
        self.render_interval = 10

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
        self.last_state = np.zeros(self.input_size)  # Initialize with correct shape
        self.last_action_log_prob = None
        self.last_value = None
        self.last_action = None

    def _build_actor_critic_models(self):
        """Builds separate Actor (policy) and Critic (value) networks mit erweiterten Layers für Stabilität."""
        # Input Layer - Now expecting a flat vector
        input_layer = Input(shape=(self.input_size,), name="flat_state_input")
        
        # Normalisierungsschicht für stabile Eingaben
        normalized = tf.keras.layers.BatchNormalization()(input_layer)
        
        # Shared layers - tieferes Netzwerk mit mehr Regularisierung
        shared_dense1 = Dense(256, activation="relu", 
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(normalized)
        dropout1 = tf.keras.layers.Dropout(0.1)(shared_dense1)
        shared_dense2 = Dense(128, activation="relu",
                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.1)(shared_dense2)

        # --- Actor Head ---
        actor_dense = Dense(64, activation="relu")(dropout2)
        # Outputs logits for each action dimension (traffic light)
        action_logits = Dense(self.output_size, activation=None, name="action_logits")(
            actor_dense
        )
        actor = Model(inputs=input_layer, outputs=action_logits)

        # --- Critic Head ---
        critic_dense = Dense(64, activation="relu")(dropout2)
        # Outputs a single value representing the estimated state value
        state_value = Dense(1, activation=None, name="state_value")(critic_dense)
        critic = Model(inputs=input_layer, outputs=state_value)

        return actor, critic

    def get_state(self, vehicles):
        """Convert vehicle group data into a flat state vector with optimierter Repräsentation.

        Args:
            vehicles: A pygame.sprite.Group containing Vehicle objects.

        Returns:
            A flat NumPy array representing the state.
        """
        state_features = []
        center_x = INTERSECTION_CENTER_X
        center_y = INTERSECTION_CENTER_Y

        if not vehicles:  # Handle empty group
            return np.zeros(self.input_size)

        # Zonen für die 4 Richtungen festlegen, statt nach Distanz zu sortieren
        zones = {
            "right": [],  # Fahrzeuge von rechts kommend
            "down": [],   # Fahrzeuge von oben kommend
            "left": [],   # Fahrzeuge von links kommend
            "up": []      # Fahrzeuge von unten kommend
        }

        # Fahrzeuge nach Zonen gruppieren
        for vehicle in vehicles:
            if vehicle.crashed:  # Skip crashed vehicles
                continue

            try:
                # Grundlegende Fahrzeugeigenschaften extrahieren
                dx = vehicle.rect.centerx - center_x
                dy = vehicle.rect.centery - center_y
                distance = math.sqrt(dx**2 + dy**2)
                angle = math.atan2(dy, dx)  # Radians from -pi to pi
                speed = min(vehicle.speed, 5.0) / 5.0  # Normalisiert auf [0,1]
                wait_time = min(vehicle.waiting_time, 50.0) / 50.0  # Normalisiert auf [0,1]
                
                # Fahrzeug in die richtige Zone einsortieren
                if hasattr(vehicle, 'direction'):
                    direction = vehicle.direction
                    if direction in zones:
                        zones[direction].append([
                            distance, 
                            speed,
                            wait_time,
                            angle
                        ])
            except AttributeError as e:
                logger.error(f"Error accessing vehicle attributes: {e}. Vehicle: {vehicle}")
                continue

        # Jede Zone sortieren und maximal (MAX_VEHICLES // 4) Fahrzeuge pro Zone nehmen
        vehicles_per_zone = MAX_VEHICLES // 4
        for direction in zones:
            # Nach Distanz sortieren (nächste zuerst)
            zones[direction].sort(key=lambda x: x[0])
            
            # Nur die nächsten N Fahrzeuge dieser Zone verwenden
            for i in range(min(len(zones[direction]), vehicles_per_zone)):
                # Alle Features des Fahrzeugs hinzufügen
                state_features.extend(zones[direction][i])
            
            # Padding für diese Zone, wenn weniger als max Fahrzeuge
            padding_needed = vehicles_per_zone - min(len(zones[direction]), vehicles_per_zone)
            if padding_needed > 0:
                state_features.extend([0.0] * (padding_needed * VEHICLE_FEATURES))

        # Konsistenzprüfung für die Vektorlänge
        if len(state_features) != self.input_size:
            logger.warning(
                f"State size mismatch. Got {len(state_features)}, expected {self.input_size}."
                f"Adding padding/truncating."
            )
            if len(state_features) < self.input_size:
                state_features.extend([0.0] * (self.input_size - len(state_features)))
            else:
                state_features = state_features[:self.input_size]

        return np.array(state_features, dtype=np.float32)

    @tf.function  # Can compile for faster inference
    def get_action_and_value(self, state):
        """Get action, log probability, and value estimate for a given state."""
        if self.actor is None or self.critic is None:
            num_active = random.randint(1, 2)
            action_indices = random.sample(range(self.output_size), num_active)
            action = [
                1.0 if i in action_indices else 0.0 for i in range(self.output_size)
            ]
            return np.array(action), 0.0, 0.0

        # Ensure state is the correct flat shape before converting to tensor
        if state.shape != (self.input_size,):
            logger.error(
                f"State shape mismatch in get_action_and_value. Expected {(self.input_size,)}, got {state.shape}. Using zeros."
            )
            # Attempt to reshape or use zeros as fallback
            try:
                state = np.reshape(state, (self.input_size,))
            except ValueError:
                state = np.zeros(self.input_size)  # Fallback to zeros if reshape fails

        state_tensor = tf.convert_to_tensor(
            [state], dtype=tf.float32
        )  # Pass as batch of 1
        action_logits = self.actor(state_tensor, training=False)
        value = self.critic(state_tensor, training=False)[0, 0]
        action_probs = tf.sigmoid(action_logits)[0]
        # Restore sampling and log_prob calculation
        action = tf.cast(
            tf.random.uniform(shape=(self.output_size,)) < action_probs,
            dtype=tf.float32,
        )
        probs_tensor = tf.clip_by_value(action_probs, 1e-10, 1.0 - 1e-10)
        log_probs_all = action * tf.math.log(probs_tensor) + (
            1.0 - action
        ) * tf.math.log(1.0 - probs_tensor)
        log_prob = tf.reduce_sum(log_probs_all)

        # Return original tuple: sampled action, log_prob, value
        return action, log_prob, value

    def calculate_step_reward(
        self,
        avg_speed,
        crashes_this_step,
        sum_sq_waiting_time,
        emissions_this_step,
        newly_crossed_count,
    ):
        """Calculate the raw reward for the current simulation step.
        Optimierte Belohnungsfunktion mit Fokus auf hohen Durchsatz (viele Fahrzeuge mit guter Geschwindigkeit).
        """
        # Aktuelle Zustandsdaten aus dem Simulationsschritt
        vehicle_count = len(self.get_current_vehicle_count())  # Gesamtzahl der Fahrzeuge in der Simulation
        
        # --- Weights (Tunable) ---
        # Statt nur die Durchschnittsgeschwindigkeit zu belohnen, belohnen wir den Gesamtdurchsatz:
        # (durchschnittliche Geschwindigkeit × Anzahl der Fahrzeuge)
        throughput_weight = 0.2            # Belohnung für Gesamtdurchsatz (Geschwindigkeit × Fahrzeuganzahl)
        waiting_penalty_weight = -0.02     # Reduziert für bessere Balance
        emission_penalty_weight = -0.5     # Reduziert für bessere Balance
        crash_penalty = -100.0             # Unverändert
        crossing_reward_weight = 15.0      # Belohnung für Fahrzeuge, die die Kreuzung überqueren

        # Clip Werte um extreme Ausreißer zu vermeiden
        sum_sq_waiting_time = min(sum_sq_waiting_time, 1000)  # Verhindert zu große Wartezeit-Strafen
        emissions_this_step = min(emissions_this_step, 50)    # Begrenzt Emissionsstrafen
        
        # Normalisierung für konsistentere Belohnungen
        normalized_speed = min(avg_speed, 5.0) / 5.0  # Normalisiert auf [0,1]
        
        reward = 0.0
        
        # --- Hauptbelohnung: Durchsatz (Geschwindigkeit × Fahrzeuganzahl) ---
        # Dies belohnt sowohl hohe Geschwindigkeit als auch die Verarbeitung vieler Fahrzeuge
        throughput = normalized_speed * vehicle_count
        reward += throughput_weight * throughput
        
        # --- Zusätzliche Belohnung für das erfolgreiche Überqueren der Kreuzung ---
        reward += crossing_reward_weight * newly_crossed_count

        # --- Strafen ---
        waiting_penalty = waiting_penalty_weight * sum_sq_waiting_time
        reward += waiting_penalty
        reward += emission_penalty_weight * emissions_this_step
        reward += crash_penalty * crashes_this_step
        
        # Reward-Clipping für stabilere Gradienten
        return np.clip(reward, -200.0, 200.0)
        
    def get_current_vehicle_count(self):
        """Hilfsmethode, um die Anzahl aktiver Fahrzeuge in der Simulation zu bestimmen."""
        import pygame
        # Globale Simulationsgruppe aus den Utility-Funktionen importieren
        from shared.utils import simulation
        
        # Zähle nur nicht-gecrasht Fahrzeuge
        active_vehicles = [v for v in simulation if hasattr(v, 'crashed') and not v.crashed]
        return active_vehicles

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
        """Perform PPO learning update using data collected since the last update."""
        if len(self.replay_buffer) < MIN_BUFFER_SIZE:
            logger.warning(
                f"Buffer size ({len(self.replay_buffer)}) is less than MIN_BUFFER_SIZE ({MIN_BUFFER_SIZE}). Skipping learning."
            )
            return  # Not enough data accumulated yet

        if not PERFORMANCE_MODE:
            print(
                f"Starting PPO Learning Update (Processing {len(self.replay_buffer)} transitions)...",
                flush=True,
            )

        # --- Use ALL data collected since the last update --- #
        # No random sampling, process the entire buffer segment
        batch = list(self.replay_buffer)

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

        # Clear the buffer segment after processing it
        self.replay_buffer.clear()
        if not PERFORMANCE_MODE:
            print(f"  Replay buffer cleared.", flush=True)

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
        current_vehicles,  # NEW Input: The group of active vehicles
        avg_speed,
        crashes_this_step,
        sum_sq_waiting_time,  # Updated parameter name
        emissions_this_step,
        newly_crossed_count,  # Updated parameter name
    ):
        """Processes one step of the simulation: calculates reward, stores transition, potentially learns.

        Args:
            current_vehicles: Pygame sprite group containing current Vehicle objects.
            avg_speed: Average speed of vehicles in the previous step.
            crashes_this_step: Number of crashes that occurred in the previous step.
            sum_sq_waiting_time: Sum of squared waiting times for vehicles.
            emissions_this_step: Total emissions from vehicles in the previous step.
            newly_crossed_count: Number of vehicles that crossed the intersection.

        Returns:
            The action (list/array of light states) determined for the *next* step.
        """
        # 1. Get state from the current vehicle group
        current_state = self.get_state(current_vehicles)

        # 2. Calculate reward based on the outcome of the *previous* step/action
        #    Note: We use metrics from the *previous* state transition leading *to* this current_state.
        #    If it's the very first step, last_state will be zeros, leading to zero reward initially.
        reward = 0.0
        if self.last_state is not None:
            # Calculate reward based on the metrics provided (which resulted from the last action)
            reward = self.calculate_step_reward(
                avg_speed,
                crashes_this_step,
                sum_sq_waiting_time,
                emissions_this_step,
                newly_crossed_count,
            )
            self.epoch_accumulated_reward += reward
            # Also update epoch summary metrics
            self.total_crashes_epoch += crashes_this_step
            # self.total_waiting_steps_epoch += waiting_vehicles_this_step # Need to rethink how to track this if needed
            self.total_emissions_epoch += emissions_this_step
            self.sum_avg_speeds_epoch += avg_speed
            self.vehicle_updates_epoch += 1  # Increment count for averaging speed

            # 3. Store the *previous* transition (state, action, reward, next_state)
            #    The 'next_state' is the 'current_state' we just calculated.
            done = False  # Assume not done unless termination condition met (e.g., max steps)
            if (
                self.last_action is not None
            ):  # Check if we actually took an action before
                self.store_transition(
                    self.last_state,
                    self.last_action,
                    reward,
                    current_state,
                    done,  # Need a proper 'done' signal if episodes end
                    self.last_action_log_prob,
                    self.last_value,
                )

        # 4. Get the action and value for the *current* state
        # Revert to expecting action, log_prob, value
        action, action_log_prob, value = self.get_action_and_value(current_state)
        next_active_lights = [bool(a > 0.5) for a in action]  # Use sampled action

        # 5. Update 'last' variables for the next iteration
        self.last_state = current_state
        # Revert to storing sampled action and its log_prob
        self.last_action = action
        self.last_action_log_prob = action_log_prob
        self.last_value = value

        # Update global step counters
        self.total_steps += 1
        self.epoch_steps += 1

        # 6. Trigger PPO learning update if enough steps have passed and buffer is full
        if (
            self.total_steps % UPDATE_EVERY_N_STEPS == 0
            and len(self.replay_buffer) >= MIN_BUFFER_SIZE
        ):
            if not PERFORMANCE_MODE:
                logger.info(f"--- Triggering PPO Learn Step {self.total_steps} ---")
            self.learn_ppo()

        # Return the action decided for the *next* simulation step
        return next_active_lights

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

    def load_models(self, filepath_prefix):
        """Loads BOTH the actor and critic models from files using a common prefix."""
        actor_path = f"{filepath_prefix}_actor.weights.h5"
        critic_path = f"{filepath_prefix}_critic.weights.h5"
        print(f"Attempting to load actor from: {actor_path}")
        print(f"Attempting to load critic from: {critic_path}")

        if os.path.exists(actor_path) and os.path.exists(critic_path):
            try:
                self.actor.load_weights(actor_path)
                print("Actor model loaded successfully.")
            except Exception as e:
                print(f"Error loading actor model weights: {e}")
                # Consider whether to proceed if only one loads

            try:
                self.critic.load_weights(critic_path)
                print("Critic model loaded successfully.")
            except Exception as e:
                print(f"Error loading critic model weights: {e}")
                # Consider whether to proceed if only one loads

        else:
            print(
                "Could not find both actor and critic weight files for the given prefix."
            )
            print(f"  Actor checked: {actor_path}")
            print(f"  Critic checked: {critic_path}")
