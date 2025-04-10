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
    PERFORMANCE_MODE,
)
from .collision_logic import check_collision

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

        self.image = None
        self.font = None
        self.width, self.height = DEFAULT_DIMS.get(
            vehicleClass, (40, 40)
        )  # Start with defaults

        # --- Load Image and Set Dimensions --- #
        if pygame.display.get_init():
            try:
                path = "images/" + direction + "/" + vehicleClass + ".png"
                loaded_image = pygame.image.load(path)
                self.image = loaded_image
                # *** Use actual image dimensions for physics and hitbox ***
                self.width = loaded_image.get_rect().width
                self.height = loaded_image.get_rect().height
                self.font = pygame.font.Font(None, 24)
                logger.debug(
                    f"Using image dimensions for {self.id} ({vehicleClass}): {self.width}x{self.height}"
                )
            except pygame.error as e:
                logger.error(
                    f"Error loading image {path} or font: {e}. Using default dimensions {self.width}x{self.height}."
                )
                # Keep default dimensions if image loading fails
                self.image = None
                self.font = None
        else:
            # If not rendering, stick with default dimensions
            logger.debug(
                f"No display; using default dimensions for {self.id} ({vehicleClass}): {self.width}x{self.height}"
            )
        # --- Image loading attempt finished --- #

        # --- Initialize self.rect *after* dimensions are set --- #
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        logger.debug(f"Initialized rect for {self.id}: {self.rect}")
        # --- End Initialize self.rect --- #

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

        # --- Refactored stop coordinate calculation (uses self.width/height) ---
        stop_calculators = {
            "right": lambda ahead: ahead.stop
            - self.width
            - stoppingGap,  # Use self.width
            "left": lambda ahead: ahead.stop + ahead.width + stoppingGap,
            "down": lambda ahead: ahead.stop
            - self.height
            - stoppingGap,  # Use self.height
            "up": lambda ahead: ahead.stop + ahead.height + stoppingGap,
        }
        if (
            self.index > 0  # Check if index is valid before accessing
            and vehicles[direction][lane][self.index - 1].crossed == 0
        ):
            vehicle_ahead = vehicles[direction][lane][self.index - 1]
            if hasattr(vehicle_ahead, "stop"):
                calculate_stop = stop_calculators.get(direction)
                if calculate_stop:
                    try:
                        # Check direction for correct attribute usage
                        if (
                            direction == "right"
                        ):  # Vehicle ahead stops based on *its* front
                            self.stop = (
                                vehicle_ahead.stop - vehicle_ahead.width - stoppingGap
                            )
                        elif (
                            direction == "left"
                        ):  # Vehicle ahead stops based on *its* front
                            self.stop = (
                                vehicle_ahead.stop + vehicle_ahead.width + stoppingGap
                            )
                        elif (
                            direction == "down"
                        ):  # Vehicle ahead stops based on *its* front
                            self.stop = (
                                vehicle_ahead.stop - vehicle_ahead.height - stoppingGap
                            )
                        elif (
                            direction == "up"
                        ):  # Vehicle ahead stops based on *its* front
                            self.stop = (
                                vehicle_ahead.stop + vehicle_ahead.height + stoppingGap
                            )
                        else:
                            self.stop = defaultStop[direction]  # Fallback

                    except AttributeError:
                        logger.warning(
                            f"AttributeError calculating stop for {self.id} behind {vehicle_ahead.id}. Using default."
                        )
                        self.stop = defaultStop[direction]
                else:
                    self.stop = defaultStop[direction]
            else:
                logger.debug(
                    f"Vehicle ahead {vehicle_ahead.id} has no 'stop' attribute yet. Using default stop for {self.id}."
                )
                self.stop = defaultStop[direction]
        else:
            self.stop = defaultStop[direction]
        # --- End Refactor ---

        # --- Refactored starting position adjustment (uses self.width/height) ---
        start_pos_adjusters = {
            "right": lambda: (
                x[direction][lane] - (self.width + stoppingGap),
                y[direction][lane],
            ),
            "left": lambda: (
                x[direction][lane] + (self.width + stoppingGap),
                y[direction][lane],
            ),
            "down": lambda: (
                x[direction][lane],
                y[direction][lane] - (self.height + stoppingGap),
            ),
            "up": lambda: (
                x[direction][lane],
                y[direction][lane] + (self.height + stoppingGap),
            ),
        }
        adjuster = start_pos_adjusters.get(direction)
        if adjuster:
            # Adjust the starting position for *this* vehicle instance
            self.x, self.y = adjuster()
            # DO NOT MODIFY GLOBAL x, y here - it caused issues
            # x[direction][lane], y[direction][lane] = adjuster()
        # --- End Refactor ---

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
            # Draw the vehicle image
            screen.blit(self.image, (self.x, self.y))

            # --- Draw the hitbox (collision rectangle) --- #
            # Use a distinct color, e.g., red, with some transparency or just outline
            hitbox_color = (255, 0, 0)  # Red
            # Draw the rectangle outline (width=1 for outline)
            pygame.draw.rect(screen, hitbox_color, self.rect, 1)
            # --- End Hitbox Drawing --- #

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

    # @cython.compile / @numba.jit (Potential Optimization Target)
    def move(
        self,
        vehicles: dict,
        active_lights: list,
        stopLines: dict,
        movingGap: int,
        simulation_group: pygame.sprite.Group,
    ):
        """
        Move the vehicle based on traffic signals, other vehicles, and turning.
        Checks for collisions before finalizing movement using the centralized check_collision.
        Refactored to use dictionaries for direction logic.

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
        old_x, old_y = self.x, self.y
        movement_occurred = False
        potential_x, potential_y = self.x, self.y

        # --- Determine Light Status ---
        direction_to_light_index = {"right": 0, "down": 1, "left": 2, "up": 3}
        my_light_index = direction_to_light_index.get(self.direction)
        is_light_green = False
        if my_light_index is not None and my_light_index < len(active_lights):
            is_light_green = active_lights[my_light_index]

        # --- Get Current Lane and Index ---
        current_lane_vehicles = vehicles[self.direction][self.lane]
        try:
            current_index = current_lane_vehicles.index(self)
        except ValueError:
            logger.warning(
                f"Vehicle {self.id} not found in its own lane list during move. Skipping."
            )
            return 0

        # --- Refactored Movement Logic ---
        def get_move_params(direction, index, current_vehicles):
            vehicle_ahead = current_vehicles[index - 1] if index > 0 else None
            crossed_check = lambda v: False  # Default
            clearance_check = lambda v, ahead: True  # Default if no vehicle ahead
            stop_condition = lambda v: False  # Default

            if direction == "right":
                crossed_check = lambda v: v.x + v.width > stopLines[direction]
                if vehicle_ahead:
                    clearance_check = lambda v, ahead: v.x + v.width < (
                        ahead.x - movingGap
                    )
                stop_condition = lambda v: v.x + v.width <= v.stop
                potential_update = lambda v: (v.x + v.speed, v.y)
            elif direction == "down":
                crossed_check = lambda v: v.y + v.height > stopLines[direction]
                if vehicle_ahead:
                    clearance_check = lambda v, ahead: v.y + v.height < (
                        ahead.y - movingGap
                    )
                stop_condition = lambda v: v.y + v.height <= v.stop
                potential_update = lambda v: (v.x, v.y + v.speed)
            elif direction == "left":
                crossed_check = lambda v: v.x < stopLines[direction]
                if vehicle_ahead:
                    clearance_check = lambda v, ahead: v.x > (
                        ahead.x + ahead.width + movingGap
                    )
                stop_condition = lambda v: v.x >= v.stop
                potential_update = lambda v: (v.x - v.speed, v.y)
            elif direction == "up":
                crossed_check = lambda v: v.y < stopLines[direction]
                if vehicle_ahead:
                    clearance_check = lambda v, ahead: v.y > (
                        ahead.y + ahead.height + movingGap
                    )
                stop_condition = lambda v: v.y >= v.stop
                potential_update = lambda v: (v.x, v.y - v.speed)

            return (
                crossed_check,
                clearance_check,
                stop_condition,
                potential_update,
                vehicle_ahead,
            )

        (
            crossed_check,
            clearance_check,
            stop_condition,
            potential_update,
            vehicle_ahead,
        ) = get_move_params(self.direction, current_index, current_lane_vehicles)

        if self.crossed == 0 and crossed_check(self):
            self.crossed = 1

        vehicle_ahead_clear = clearance_check(self, vehicle_ahead)
        should_move = (
            stop_condition(self) or self.crossed == 1 or is_light_green
        ) and vehicle_ahead_clear

        # --- Pre-move Collision Check --- #
        if should_move:
            # Create a temporary vehicle state for collision checking
            temp_vehicle = pygame.sprite.Sprite()  # Lightweight temp object
            temp_vehicle.x = potential_update(self)[0]
            temp_vehicle.y = potential_update(self)[1]
            temp_vehicle.width = self.width
            temp_vehicle.height = self.height
            temp_vehicle.direction = self.direction
            temp_vehicle.lane = self.lane
            temp_vehicle.crashed = False  # Important: temp isn't crashed yet
            temp_vehicle.id = self.id  # For logging in collision check

            potential_colliders = simulation_group  # Check against all others

            for other_vehicle in potential_colliders:
                if other_vehicle is self:  # Don't check against self
                    continue

                # Call the centralized collision logic
                if check_collision(temp_vehicle, other_vehicle):
                    self.crashed = True
                    other_vehicle.crashed = True  # Mark both as crashed
                    crashed_this_step = 1
                    # Logger message is now inside check_collision for context
                    should_move = False  # Prevent movement if collision detected
                    break  # Stop checking after first collision involving this vehicle

        # --- Finalize Movement (if no collision occurred) --- #
        if should_move:
            # Apply the potential update calculated earlier
            self.x, self.y = potential_update(self)
            movement_occurred = True

            # --- Update Rect --- #
            # Crucial to update rect *after* position is finalized
            self.rect.topleft = (self.x, self.y)
            # --- End Update Rect --- #

        # --- Update Waiting Time / Acceleration / Emission (Refactored) --- #
        self.accelerated = False
        self.decelerated = False
        emission_multiplier = 0.0  # Default

        if not movement_occurred:
            if not self.crashed:
                # Increment waiting time if not crashed and didn't move
                if self.id not in waiting_times[self.direction]:
                    waiting_times[self.direction][self.id] = 0
                waiting_times[self.direction][self.id] += 1
                self.waiting_time += 1

                # Determine if stopped or truly idle
                if (
                    old_x != self.x or old_y != self.y
                ):  # Could have moved but didn't (e.g., collision)
                    self.decelerated = True
                    emission_multiplier = 0.1  # Base waiting emission multiplier
                else:  # Truly stopped/idle
                    emission_multiplier = 0.05  # Lower idle emission multiplier
            # Else: Crashed vehicles don't wait or accrue emissions here

        elif self.waiting_time > 0:  # Moved after waiting
            self.accelerated = True
            emission_multiplier = 1.5  # Acceleration emission multiplier
            self.waiting_time = 0  # Reset waiting time
        else:  # Moved without prior waiting (cruising)
            emission_multiplier = 1.0  # Cruising emission multiplier

        # Apply emission calculation if not crashed
        if not self.crashed and emission_multiplier > 0:
            base_emission_factor = emission_factors.get(self.vehicleClass, 1.0)
            self.emission += base_emission_factor * emission_multiplier
            # Note: self.emission accumulates over time. The update_emission method
            # seems to calculate per-step emission, which might be redundant now?
            # Consider if self.emission should store total or per-step.

        return crashed_this_step


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
