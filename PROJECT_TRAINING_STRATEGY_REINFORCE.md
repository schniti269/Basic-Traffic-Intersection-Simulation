# Traffic Simulation RL Training Strategy: REINFORCE

## 1. Business Understanding

*   **Objective:** Optimize traffic light control in a simulated intersection to reduce congestion (waiting time), prevent crashes, minimize emissions, and maximize vehicle throughput (average speed).
*   **Problem:** The current training method relies on aggregated epoch-level rewards, leading to poor credit assignment (difficulty attributing outcomes to specific actions) and suboptimal learning.
*   **Proposed Solution:** Implement the REINFORCE policy gradient algorithm, a standard Reinforcement Learning technique, for more effective learning based on per-step feedback and better credit assignment.

## 2. Data Understanding

*   **State (S\\_t):** A `(4, 20, 5)` NumPy array derived from sensor data (vehicle distance, speed, type, acceleration, waiting time) in 4 zones. Processed by the `get_state` method in `neural_model_01.py`. Captures the situation *before* an action is taken.
*   **Action (A\\_t):** A multi-label binary vector (length 4) indicating active traffic lights (e.g., `[1, 0, 1, 0]` means lights 0 and 2 are active). Derived by thresholding the model's output probabilities (>= 0.5).
*   **Model Output (Policy \\(\\pi_\\theta(A|S)\\)):** The neural network outputs probabilities `p = [p_0, p_1, p_2, p_3]` for each light being active, using a sigmoid activation function. This represents the policy.
*   **Reward (R\\_{t+1}):** Calculated *after* action \(A_t\) is taken and the simulation advances one step. It reflects the immediate outcome of the action and the resulting state \(S_{t+1}\). The **per-step reward function** (`calculate_step_reward`) computes this scalar value based on metrics *from the completed step*:
    *   Average vehicle speed *during the step*.
    *   Number of vehicles waiting *at the end of the step*.
    *   Number of crashes *detected during the step*.
    *   Total emissions *generated during the step*.
    *   (Optional) Small negative penalty for changing light states frequently.

## 3. Data Preparation

*   The existing `get_state` function handles the conversion of raw simulation observations into the normalized, fixed-size state representation required by the model. This appears sufficient for the REINFORCE implementation.

## 4. Modeling (REINFORCE Algorithm)

*   **Algorithm Choice:** REINFORCE (Monte Carlo Policy Gradient). This is suitable as the model directly parameterizes the policy (outputs action probabilities).
*   **Objective:** Maximize the expected total discounted return \( J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [ \sum_{t=0}^{T} \gamma^t R_{t+1} ] \), where \( \tau \) is a trajectory (an epoch of states, actions, rewards), \( \pi_\theta \) is the policy parameterized by network weights \( \theta \), \( T \) is the epoch length, and \( \gamma \) is the discount factor (e.g., 0.99) valuing immediate rewards more than future ones.
*   **Return Calculation:** At the end of each epoch (trajectory), calculate the discounted return \( G_t \) for each time step \( t \) within that epoch:
    \[ G_t = \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1} \]
    This is calculated efficiently by working backwards from the end of the epoch.
*   **Policy Gradient & Loss Function:** The core idea is to adjust \( \theta \) to increase the likelihood of actions that lead to high returns. The policy gradient theorem gives us the direction: \( \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [ \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(A_t|S_t) ] \). For gradient *descent* using optimizers like Adam, we define a loss function whose negative gradient matches this. The loss for a single trajectory (epoch) is:
    \[ L_{epoch}(\theta) = - \sum_{t=0}^{T} G_t \log \pi_\theta(A_t|S_t) \]
*   **Log-Probability Term (\( \log \pi_\theta(A_t|S_t) \)):** Since we have a multi-label action \( A_t = [a_0, a_1, a_2, a_3] \) (where \(a_i \in \{0, 1\}\) is the actual state of light \(i\) chosen at step \(t\)) and sigmoid outputs \( p_t = [p_{0,t}, ..., p_{3,t}] \), the log-probability is the sum of the log-probabilities of each independent Bernoulli trial (each light):
    \[ \log \pi_\theta(A_t|S_t) = \sum_{i=0}^{3} [ a_{i,t} \log p_{i,t} + (1 - a_{i,t}) \log(1 - p_{i,t}) ] \]
    This is equivalent to the negative binary cross-entropy between the chosen action \( A_t \) and the predicted probabilities \( p_t \).
*   **Final Loss for Training:** Combining the above, the loss computed for the epoch is:
    \[ L_{epoch}(\theta) = - \sum_{t=0}^{T} G_t \left( \sum_{i=0}^{3} [ a_{i,t} \log p_{i,t} + (1 - a_{i,t}) \log(1 - p_{i,t}) ] \right) \]
*   **Training Update:** We will use TensorFlow's `tf.GradientTape` to automatically compute the gradients \( \nabla_\theta L_{epoch}(\theta) \) and then apply these gradients to update the model's weights \( \theta \) using the existing Adam optimizer instance.

## 5. Evaluation

*   **Metrics:** Continue tracking epoch-level performance using aggregated metrics (sum of step rewards for total reward, average speed, total crashes, total waiting steps, total emissions). Monitor the trend of these metrics across epochs to assess learning progress.
*   **Qualitative Assessment:** Utilize the simulation's rendering capability (controlled by `render_interval`) to visually inspect the traffic light control policy learned by the agent. Observe if the patterns seem more intelligent or efficient compared to the previous method.

## 6. Implementation Plan (`neural_model_01.py` Refactoring)

1.  **`__init__`:**
    *   Add `discount_factor` (gamma, e.g., 0.99) parameter.
    *   Initialize new lists to store per-step data for the epoch: `self.epoch_rewards`, `self.epoch_actual_actions`, `self.epoch_log_probs`.
    *   Ensure the optimizer (`Adam`) is stored as an instance variable (e.g., `self.optimizer`) so it can be used with `GradientTape`. Modify `_build_model` or `__init__` accordingly.
2.  **`remember_step`:** Modify to accept and store `state`, `action_probs`, `actual_action` (the thresholded boolean list), and `reward` into the corresponding epoch lists.
3.  **`calculate_step_reward`:**
    *   Create this new method within `NeuralTrafficController`.
    *   It will receive the necessary metrics available in `update` (e.g., `avg_speed`, `new_crashes_this_step`, `waiting_vehicles`, `emissions_this_step`).
    *   Implement the logic to calculate and return a scalar reward `R_{t+1}` based on these inputs. (An initial version will be created).
4.  **`update` Method:**
    *   Get state \(S_t\). Get action probabilities \(p_t\) and determine actual action \(A_t\) (e.g., `active_lights`).
    *   *Crucially, the simulation main loop must now execute the vehicle movement based on \(A_t\) and calculate the resulting metrics (crashes, waiting vehicles, emissions, avg speed for step t+1).* These metrics are then passed back into the controller's `update` method.
    *   Call `calculate_step_reward` using the metrics from the *just completed* simulation step to get \(R_{t+1}\).
    *   Calculate \(\log \pi_\theta(A_t|S_t)\) using \(p_t\) and \(A_t\).
    *   Call `remember_step` to store the `(S_t, A_t, R_{t+1}, \log \pi_\theta(A_t|S_t))` tuple.
    *   The method returns \(A_t\) which the main loop *should have already used* for the movement calculation.
5.  **`train_epoch` Method:**
    *   *Completely rewrite* this method.
    *   Implement the backward calculation of discounted returns `G_t` using `self.epoch_rewards` and `self.discount_factor`.
    *   Convert stored epoch lists (`epoch_states`, `epoch_action_probs`, `epoch_actual_actions`, `epoch_rewards`, `epoch_log_probs`) and calculated returns `G_t` into TensorFlow tensors.
    *   Use `tf.GradientTape` to watch the model's trainable variables.
    *   Inside the tape's context, perform a forward pass with `self.epoch_states` to get predicted probabilities (needed for loss calculation, although we already have `epoch_action_probs` stored - using stored ones might be more efficient). *Correction:* We need the probabilities corresponding to the *stored states* to compute the gradient correctly. We should re-compute probabilities from `epoch_states` inside the `GradientTape` context.
    *   Calculate the REINFORCE loss using the formula defined in Section 4, employing `tf.losses.binary_crossentropy` (or manual calculation) with the re-computed probabilities, the stored `epoch_actual_actions`, and the calculated `G_t`. Remember the negative sign and the weighting by `G_t`.
    *   Compute gradients: `grads = tape.gradient(loss, self.model.trainable_variables)`.
    *   Apply gradients: `self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))`.
6.  **`end_epoch` Method:**
    *   Call the new `train_epoch`.
    *   Modify the summary printout: report the *sum* of step rewards for the epoch as the 'Total Reward'.
    *   Crucially, clear the epoch data storage lists: `self.epoch_states`, `self.epoch_actual_actions`, `self.epoch_rewards`, `self.epoch_log_probs`.

This plan provides a clear roadmap for transitioning to the REINFORCE algorithm, incorporating fixes for collision logic and clarifying the flow of state, action, and reward calculation. 