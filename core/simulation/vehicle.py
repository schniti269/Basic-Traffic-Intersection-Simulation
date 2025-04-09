import pygame
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.utils import (
    speeds,
    x,
    y,
    vehicles,
    waiting_times,
    stoppingGap,
    movingGap,
    stopLines,
    defaultStop,
    emission_factors,
    simulation,
    logger,
    directionNumbers,
)

# Default dimensions when rendering is off
DEFAULT_DIMS = {
    "car": (40, 40),
    "bus": (60, 60),
    "truck": (60, 60),
    "bike": (20, 20),
}


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.id = f"{direction}_{lane}_{len(vehicles[direction][lane])}"
        self.waiting_time = 0
        self.accelerated = False
        self.decelerated = False
        self.crashed = False
        self.emission = 0

        # Initialize waiting time for this vehicle ID
        if self.id not in waiting_times[direction]:
            waiting_times[direction][self.id] = 0

        # Correctly assign index *before* appending
        self.index = len(vehicles[direction][lane])
        vehicles[direction][lane].append(
            self
        )  # Append self *after* index is calculated

        # Set default dimensions
        self.width, self.height = DEFAULT_DIMS.get(vehicleClass, (40, 40))
        self.image = None
        self.font = None  # Initialize font as None

        # Only load images and fonts if Pygame display is initialized (rendering is active)
        if pygame.display.get_init():
            try:
                path = "images/" + direction + "/" + vehicleClass + ".png"
                self.image = pygame.image.load(path)
                self.width = self.image.get_rect().width
                self.height = self.image.get_rect().height
                # Initialize font only when needed for rendering
                self.font = pygame.font.Font(None, 24)
            except pygame.error as e:
                logger.error(
                    f"Error loading image {path} or font: {e}. Using default dimensions."
                )
                self.image = None  # Ensure image is None if loading failed
                self.font = None

        # Calculate stop coordinate based on vehicle ahead or default
        if (
            len(vehicles[direction][lane]) > 1
            and vehicles[direction][lane][self.index - 1].crossed == 0
        ):
            vehicle_ahead = vehicles[direction][lane][self.index - 1]
            if direction == "right":
                self.stop = vehicle_ahead.stop - vehicle_ahead.width - stoppingGap
            elif direction == "left":
                self.stop = vehicle_ahead.stop + vehicle_ahead.width + stoppingGap
            elif direction == "down":
                self.stop = vehicle_ahead.stop - vehicle_ahead.height - stoppingGap
            elif direction == "up":
                self.stop = vehicle_ahead.stop + vehicle_ahead.height + stoppingGap
        else:
            self.stop = defaultStop[direction]

        # Adjust starting positions for new vehicles
        if direction == "right":
            temp = self.width + stoppingGap
            x[direction][lane] -= temp
        elif direction == "left":
            temp = self.width + stoppingGap
            x[direction][lane] += temp
        elif direction == "down":
            temp = self.height + stoppingGap
            y[direction][lane] -= temp
        elif direction == "up":
            temp = self.height + stoppingGap
            y[direction][lane] += temp

        simulation.add(self)

    def load_image_if_needed(self):
        """Loads the Pygame image and font if they aren't already loaded and rendering is active."""
        if self.image is None and pygame.display.get_init():
            logger.debug(f"Loading image for {self.id} due to rendering start.")
            try:
                path = "images/" + self.direction + "/" + self.vehicleClass + ".png"
                self.image = pygame.image.load(path)
                self.width = self.image.get_rect().width
                self.height = self.image.get_rect().height
                # Ensure font is also loaded if missing
                if self.font is None:
                    self.font = pygame.font.Font(None, 24)
            except pygame.error as e:
                logger.error(
                    f"Error loading image {path} or font: {e}. Using default dimensions."
                )
                self.image = None
                self.font = None

    def update_emission(self):
        """
        Calculate emissions based on speed and vehicle type.
        Returns the emission value for this step.
        """
        emission = 0
        if self.speed < 5:  # Higher emissions at low speeds/idling
            if self.vehicleClass == "car":
                emission = 0.15
            elif self.vehicleClass == "bus":
                emission = 0.25
            elif self.vehicleClass == "truck":
                emission = 0.30
            elif self.vehicleClass == "bike":
                emission = 0.05
        else:  # Lower emissions at cruising speed
            if self.vehicleClass == "car":
                emission = 0.05
            elif self.vehicleClass == "bus":
                emission = 0.10
            elif self.vehicleClass == "truck":
                emission = 0.15
            elif self.vehicleClass == "bike":
                emission = 0.01

        # Scale by acceleration/deceleration
        if self.accelerated:
            emission *= 1.2
        elif self.decelerated:
            emission *= 0.8

        self.emission = emission  # Store on vehicle instance if needed later
        return emission  # Return the value for this step

    def render(self, screen):
        # Only render if image exists and Pygame display is active
        if self.image and pygame.display.get_init():
            # Draw the vehicle
            screen.blit(self.image, (self.x, self.y))

            # Only show waiting time if the vehicle is waiting and font is available
            if self.waiting_time > 0 and not self.crossed and self.font:
                # Create text surface with waiting time
                waiting_text = f"{int(self.waiting_time)}s"
                text_surface = self.font.render(
                    waiting_text, True, (255, 255, 255)
                )  # White text

                # Calculate position for text (centered above vehicle)
                text_x = self.x + (self.width - text_surface.get_rect().width) // 2
                text_y = self.y - text_surface.get_rect().height - 5

                # Draw black background for better visibility
                padding = 2
                bg_rect = pygame.Rect(
                    text_x - padding,
                    text_y - padding,
                    text_surface.get_rect().width + 2 * padding,
                    text_surface.get_rect().height + 2 * padding,
                )
                pygame.draw.rect(screen, (0, 0, 0), bg_rect)  # Black background

                # Draw the text
                screen.blit(text_surface, (text_x, text_y))

    def move(self, vehicles, active_lights, stopLines, movingGap, simulation_group):
        """
        Move the vehicle based on traffic signals, other vehicles, and turning.
        Checks for collisions before finalizing movement.

        Args:
            vehicles (dict): Dictionary containing all vehicles, keyed by direction and lane.
            active_lights (list): List of booleans indicating active traffic lights.
            stopLines (dict): Dictionary of stop line coordinates.
            movingGap (int): Minimum gap required between moving vehicles.
            simulation_group (pygame.sprite.Group): The sprite group containing all vehicles.

        Returns:
            int: 1 if a new crash occurred involving this vehicle this step, 0 otherwise.
        """
        crashed_this_step = 0
        old_x = self.x
        old_y = self.y
        movement_occurred = False
        should_move = False
        potential_x = self.x
        potential_y = self.y

        # --- Determine if vehicle *wants* to move (based on lights, vehicle ahead) ---
        direction_to_light_index = {"right": 0, "down": 1, "left": 2, "up": 3}
        my_light_index = direction_to_light_index.get(self.direction)
        is_light_green = False
        if my_light_index is not None and my_light_index < len(active_lights):
            is_light_green = active_lights[my_light_index]

        current_lane_vehicles = vehicles[self.direction][self.lane]
        try:
            current_index = current_lane_vehicles.index(self)
        except ValueError:
            logger.warning(
                f"Vehicle {self.id} not found in its own lane list during move. Skipping."
            )
            return 0

        if self.direction == "right":
            if self.crossed == 0 and self.x + self.width > stopLines[self.direction]:
                self.crossed = 1
            vehicle_ahead_clear = current_index == 0 or self.x + self.width < (
                current_lane_vehicles[current_index - 1].x - movingGap
            )
            should_move = (
                self.x + self.width <= self.stop or self.crossed == 1 or is_light_green
            ) and vehicle_ahead_clear
            if should_move:
                potential_x += self.speed

        elif self.direction == "down":
            if self.crossed == 0 and self.y + self.height > stopLines[self.direction]:
                self.crossed = 1
            vehicle_ahead_clear = current_index == 0 or self.y + self.height < (
                current_lane_vehicles[current_index - 1].y - movingGap
            )
            should_move = (
                self.y + self.height <= self.stop or self.crossed == 1 or is_light_green
            ) and vehicle_ahead_clear
            if should_move:
                potential_y += self.speed

        elif self.direction == "left":
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
            vehicle_ahead_clear = current_index == 0 or self.x > (
                current_lane_vehicles[current_index - 1].x
                + current_lane_vehicles[current_index - 1].width
                + movingGap
            )
            should_move = (
                self.x >= self.stop or self.crossed == 1 or is_light_green
            ) and vehicle_ahead_clear
            if should_move:
                potential_x -= self.speed

        elif self.direction == "up":
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
            vehicle_ahead_clear = current_index == 0 or self.y > (
                current_lane_vehicles[current_index - 1].y
                + current_lane_vehicles[current_index - 1].height
                + movingGap
            )
            should_move = (
                self.y >= self.stop or self.crossed == 1 or is_light_green
            ) and vehicle_ahead_clear
            if should_move:
                potential_y -= self.speed

        # --- Check for Collisions *before* moving --- #
        if should_move:
            # Create potential new rect for collision checking
            potential_rect = pygame.Rect(
                potential_x, potential_y, self.width, self.height
            )

            # Check against all other vehicles in the simulation group
            for other_vehicle in simulation_group:
                if (
                    other_vehicle is self or other_vehicle.crashed
                ):  # Skip self and already crashed
                    continue

                other_rect = pygame.Rect(
                    other_vehicle.x,
                    other_vehicle.y,
                    other_vehicle.width,
                    other_vehicle.height,
                )

                if potential_rect.colliderect(other_rect):
                    # Collision detected!
                    self.crashed = True
                    other_vehicle.crashed = True
                    crashed_this_step = (
                        1  # Mark that a crash happened this step involving self
                    )
                    logger.warning(
                        f"Collision! Vehicle {self.id} ({self.direction}) at ({potential_x:.1f},{potential_y:.1f}) and Vehicle {other_vehicle.id} ({other_vehicle.direction}) at ({other_vehicle.x:.1f},{other_vehicle.y:.1f})"
                    )
                    should_move = False  # Prevent movement due to collision
                    break  # Stop checking for this vehicle once a crash occurs

        # --- Finalize Movement (if no collision occurred) --- #
        if should_move:
            self.x = potential_x
            self.y = potential_y
            movement_occurred = True

        # --- Update Waiting Time / Acceleration / Emission --- #
        if not movement_occurred:
            # If vehicle didn't move (due to red light, vehicle ahead, or collision), increment waiting time
            # Only count as waiting if not already crashed
            if not self.crashed:
                if self.id not in waiting_times[self.direction]:
                    waiting_times[self.direction][self.id] = 0
                waiting_times[self.direction][self.id] += 1
                self.waiting_time += 1

            # Consider it deceleration if position *could* have changed but didn't
            if old_x != self.x or old_y != self.y:
                self.decelerated = True
                self.accelerated = False
                # Only add emission if not crashed
                if not self.crashed:
                    emission_amount = (
                        emission_factors.get(self.vehicleClass, 1.0) * 0.1
                    )  # Base waiting emission
                    self.emission += emission_amount
            else:
                # Truly stopped
                self.decelerated = False
                self.accelerated = False
                # Add idle emission if not crashed
                if not self.crashed:
                    emission_amount = (
                        emission_factors.get(self.vehicleClass, 1.0) * 0.05
                    )  # Lower idle emission
                    self.emission += emission_amount

        elif self.waiting_time > 0:  # Moved after waiting
            self.accelerated = True
            self.decelerated = False
            # Add acceleration emission if not crashed
            if not self.crashed:
                emission_amount = (
                    emission_factors.get(self.vehicleClass, 1.0) * 1.5
                )  # Higher emission for acceleration
                self.emission += emission_amount
            self.waiting_time = 0
        else:  # Moved without prior waiting (cruising)
            self.accelerated = False
            self.decelerated = False
            # Add cruising emission if not crashed
            if not self.crashed:
                emission_amount = (
                    emission_factors.get(self.vehicleClass, 1.0) * 1.0
                )  # Base cruising emission
                self.emission += emission_amount

        return (
            crashed_this_step  # Return 1 if self was involved in a new crash this step
        )


# Global variable for the background thread
vehicle_generation_thread = None


def generateVehicles():
    """Background thread function to generate vehicles."""
    global vehicles, simulation, x, y, stoppingGap  # Ensure globals are accessible
    # Add imports needed within the thread context if not already global
    import random
    import time
    from shared.utils import vehicleTypes, directionNumbers

    # sys.path modification might be needed here too if utils isn't found
    # Potentially simpler: Pass necessary configs/globals as arguments to the thread

    while True:
        # Correctly choose a random vehicle type key
        vehicle_type_key = random.choice(
            list(vehicleTypes)
        )  # Choose a key (0, 1, 2, or 3)
        lane_number = random.randint(1, 2)
        will_turn = 0
        if lane_number == 1:
            will_turn = random.randint(0, 1)

        # Generate direction number (0 to 3)
        direction_number = random.randint(0, 3)
        direction = directionNumbers[direction_number]

        # Check if space is available before creating
        can_add = True
        if len(vehicles[direction][lane_number]) > 0:
            last_vehicle = vehicles[direction][lane_number][-1]
            if direction == "right":
                if (
                    last_vehicle.x
                    < x[direction][lane_number] - last_vehicle.width - stoppingGap
                ):
                    can_add = False
            elif direction == "left":
                if (
                    last_vehicle.x
                    > x[direction][lane_number] + last_vehicle.width + stoppingGap
                ):
                    can_add = False
            elif direction == "down":
                if (
                    last_vehicle.y
                    < y[direction][lane_number] - last_vehicle.height - stoppingGap
                ):
                    can_add = False
            elif direction == "up":
                if (
                    last_vehicle.y
                    > y[direction][lane_number] + last_vehicle.height + stoppingGap
                ):
                    can_add = False

        if can_add:
            vehicle_class_name = vehicleTypes[
                vehicle_type_key
            ]  # Get the name ('car', 'bus', etc.)
            logger.debug(
                f"Generated Vehicle: Type={vehicle_class_name}, Lane={lane_number}, Dir={direction}, Turn={will_turn}"  # Log the name
            )
            vehicle = Vehicle(
                lane_number,
                vehicle_class_name,  # Pass the name to the constructor
                direction_number,
                direction,
                will_turn,
            )
            simulation.add(vehicle)
        else:
            logger.debug(
                f"Skipped vehicle generation for {direction} lane {lane_number} due to space."
            )

        time.sleep(random.uniform(0.5, 1.5))  # Randomize generation frequency


# --- Start the background thread automatically --- #
# Removed - This should be called from the main scripts (train/run_model)
# def start_vehicle_generation_thread():
#     global vehicle_generation_thread
#     if vehicle_generation_thread is None or not vehicle_generation_thread.is_alive():
#         vehicle_generation_thread = threading.Thread(
#             name="generateVehicles", target=generateVehicles, daemon=True
#         )
#         vehicle_generation_thread.start()
#         logger.info("Vehicle generation thread started.")
#
# start_vehicle_generation_thread()
