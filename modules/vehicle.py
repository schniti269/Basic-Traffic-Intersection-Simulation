import random
import math
import pygame
import logging
import numpy as np
import uuid

logger = logging.getLogger("Vehicle")

# Define vehicle types and their properties: (width, length)
VEHICLE_SIZES = {
    "car": (40, 60),
    "bus": (50, 120),
    "truck": (50, 100),
    "bike": (20, 40),
}

# Define vehicle colors
VEHICLE_COLORS = {
    "car": [
        (200, 0, 0),
        (0, 200, 0),
        (0, 0, 200),
        (200, 200, 0),
        (200, 0, 200),
        (0, 200, 200),
    ],
    "bus": [(200, 100, 0), (100, 200, 0)],
    "truck": [(100, 100, 100), (150, 150, 150)],
    "bike": [(50, 50, 50), (100, 100, 100)],
}

# Define vehicle speeds (pixels per frame)
VEHICLE_SPEEDS = {
    "car": {"min": 2.0, "max": 3.0},
    "bus": {"min": 1.5, "max": 2.0},
    "truck": {"min": 1.5, "max": 2.2},
    "bike": {"min": 2.5, "max": 3.5},
}

# Define emissions per vehicle type (arbitrary units)
VEHICLE_EMISSIONS = {"car": 1.0, "bus": 2.5, "truck": 3.0, "bike": 0.2}


class Vehicle(pygame.sprite.Sprite):
    """Vehicle class for traffic simulation"""

    def __init__(self, lane, vehicleClass, direction_number, direction, controller):
        pygame.sprite.Sprite.__init__(self)

        # Vehicle identification
        self.id = uuid.uuid4().hex[:8]  # Unique vehicle ID
        self.vehicleClass = vehicleClass
        self.lane = lane
        self.direction_number = direction_number
        self.direction = direction
        self.controller = controller

        # Position and movement
        self.x = controller.x[direction][lane]
        self.y = controller.y[direction][lane]
        self.crossed = 0
        self.waiting_time = 0
        self.acceleration = 0.05
        self.deceleration = 0.1

        # Detection box size
        width, height = VEHICLE_SIZES[vehicleClass]
        self.width = width
        self.height = height

        # Movement parameters
        self.speed = controller.speeds[vehicleClass]
        self.initial_speed = self.speed
        self.max_speed = self.speed * 1.5
        self.stop = controller.defaultStop[direction]

        # Emissions and state
        self.emissions = 0  # CO2 emissions counter
        self.time_in_system = 0  # Time spent in the system
        self.distance_traveled = 0  # Distance traveled in the system
        self.crashed = False

        # Collisions
        self.collision_tolerance = 5  # Tolerance for collision detection in pixels

        # Pygame sprite setup
        try:
            # Create a simple rect for rendering (can be replaced with image)
            self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
            self.color = random.choice(
                VEHICLE_COLORS.get(vehicleClass, VEHICLE_COLORS["car"])
            )

            # Create surface with per-pixel alpha for a nicer look
            self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

            # Draw vehicle with slightly rounded corners
            pygame.draw.rect(
                self.image, self.color, (0, 0, self.width, self.height), border_radius=3
            )

            # Adjust rect for position
            self.rect.topleft = (self.x, self.y)

            # Add to controller's vehicles dictionary
            controller.vehicles[direction][lane].append(self)

            # Add to spatial hash
            if controller.use_spatial_hash:
                controller.spatial_hash.add_object(self)

            logger.debug(
                f"Vehicle {self.id} ({vehicleClass}) created at ({self.x}, {self.y})"
            )

        except Exception as e:
            logger.error(f"Error initializing vehicle: {e}")

    def move(self):
        """Move the vehicle according to traffic rules and physics"""
        if self.crashed:
            return

        # Add to the time in the system
        self.time_in_system += 1

        # Previous position for calculating distance
        prev_x, prev_y = self.x, self.y

        # Detect signal and update movement
        self._update_movement()

        # Update position
        if self.direction == "right":
            self._move_right()
        elif self.direction == "down":
            self._move_down()
        elif self.direction == "left":
            self._move_left()
        elif self.direction == "up":
            self._move_up()

        # Calculate distance traveled in this step
        dx = self.x - prev_x
        dy = self.y - prev_y
        dist = math.sqrt(dx * dx + dy * dy)
        self.distance_traveled += dist

        # Calculate emissions based on distance and vehicle type
        from modules.traffic import CO2_EMISSIONS

        emission_factor = CO2_EMISSIONS[self.vehicleClass]
        self.emissions += emission_factor * dist / 1000  # Convert to km

        # Update controller's emission counter
        self.controller.emission_counts[self.direction] += emission_factor * dist / 1000

        # Update the vehicle's rect position
        self.rect.topleft = (self.x, self.y)

        # Update in spatial hash
        if self.controller.use_spatial_hash and not self.controller.performance_mode:
            self.controller.spatial_hash.add_object(self)

        # Check for collisions
        if not self.controller.performance_mode:
            self._check_collisions()

    def _update_movement(self):
        """Update waiting status and speed based on signal and other vehicles"""
        if self.crossed == 0 and self._is_at_signal():  # Check if at traffic signal
            if (
                self.direction_number == self.controller.currentGreen
                and self.controller.currentYellow == 0
            ):
                self.crossed = 1
                self.waiting_time = 0
            else:
                # Need to stop at signal
                self._approach_stop()
                # Increment waiting time
                self.waiting_time += 1

                # Track waiting vehicle in controller for metrics
                _id = str(self.id)
                if _id not in self.controller.waiting_times[self.direction]:
                    self.controller.waiting_times[self.direction][_id] = 0
                self.controller.waiting_times[self.direction][_id] += 1

        # Otherwise move normally and check for vehicle ahead
        else:
            # If crossed the signal, restore normal speed
            if self.crossed == 1:
                self._restore_speed()

            # Check for vehicles ahead to avoid collisions
            if not self.controller.performance_mode:
                self._avoid_collision()

    def _is_at_signal(self):
        """Check if the vehicle is at a signal stop line"""
        if self.direction == "right":
            return self.x + self.width > self.stop
        elif self.direction == "down":
            return self.y + self.height > self.stop
        elif self.direction == "left":
            return self.x < self.stop
        elif self.direction == "up":
            return self.y < self.stop
        return False

    def _approach_stop(self):
        """Slow down and approach the stop line"""
        # Gradually reduce speed as approaching stop
        distance_to_stop = abs(self._distance_to_stop())

        if distance_to_stop < 10:  # Very close to stop
            self.speed = 0
        elif distance_to_stop < 50:  # Approaching stop
            # Smoothly reduce speed
            self.speed = max(0, self.speed - self.deceleration)
        elif self.speed < self.initial_speed:
            # Gradually increase if we're further away
            self.speed = min(self.initial_speed, self.speed + self.acceleration)

    def _distance_to_stop(self):
        """Calculate distance to stop line"""
        if self.direction == "right":
            return self.stop - (self.x + self.width)
        elif self.direction == "down":
            return self.stop - (self.y + self.height)
        elif self.direction == "left":
            return self.x - self.stop
        elif self.direction == "up":
            return self.y - self.stop

    def _restore_speed(self):
        """Restore speed after crossing signal"""
        if self.speed < self.initial_speed:
            self.speed = min(self.initial_speed, self.speed + self.acceleration)

    def _move_right(self):
        """Move vehicle to the right"""
        # Move vehicle
        self.x += self.speed

        # Check if we're beyond the simulation area
        if self.x > 1400:
            self._exit_simulation()

    def _move_down(self):
        """Move vehicle down"""
        # Move vehicle
        self.y += self.speed

        # Check if we're beyond the simulation area
        if self.y > 800:
            self._exit_simulation()

    def _move_left(self):
        """Move vehicle to the left"""
        # Move vehicle
        self.x -= self.speed

        # Check if we're beyond the simulation area
        if self.x < 0:
            self._exit_simulation()

    def _move_up(self):
        """Move vehicle up"""
        # Move vehicle
        self.y -= self.speed

        # Check if we're beyond the simulation area
        if self.y < 0:
            self._exit_simulation()

    def _exit_simulation(self):
        """Remove vehicle when it exits the simulation area"""
        # Increment crossed count
        self.controller.vehicles[self.direction]["crossed"] += 1

        # Update controller stats
        # (can add more tracking here)

        # Remove from the lane
        if self in self.controller.vehicles[self.direction][self.lane]:
            self.controller.vehicles[self.direction][self.lane].remove(self)

        # Remove from sprites
        self.kill()

        logger.debug(f"Vehicle {self.id} exited the simulation via {self.direction}")

    def _avoid_collision(self):
        """Check for vehicles ahead and adjust speed to avoid collisions"""
        # Check for vehicles ahead in the same lane
        lane_vehicles = self.controller.vehicles[self.direction][self.lane]

        # Find index of this vehicle
        try:
            index = lane_vehicles.index(self)
        except ValueError:
            return  # Vehicle not in list, probably exiting

        # If not the first vehicle in lane, check if we need to slow down
        if index > 0:
            vehicle_ahead = lane_vehicles[index - 1]

            # Calculate distance between vehicles
            distance = self._calculate_distance(vehicle_ahead)

            # Adjust speed based on distance
            safe_distance = self.controller.movingGap + max(self.width, self.height) / 2

            if distance < safe_distance:
                # Too close - decelerate
                self.speed = max(0, self.speed - self.deceleration * 2)
            elif distance < safe_distance * 2:
                # Approaching - gradual deceleration
                self.speed = max(0, self.speed - self.deceleration)
            elif self.speed < self.initial_speed:
                # Far enough - can accelerate back to normal speed
                self.speed = min(self.initial_speed, self.speed + self.acceleration)

    def _calculate_distance(self, other_vehicle):
        """Calculate distance to another vehicle"""
        if self.direction == "right":
            return other_vehicle.x - (self.x + self.width)
        elif self.direction == "down":
            return other_vehicle.y - (self.y + self.height)
        elif self.direction == "left":
            return (self.x) - (other_vehicle.x + other_vehicle.width)
        elif self.direction == "up":
            return (self.y) - (other_vehicle.y + other_vehicle.height)
        return float("inf")  # Default to a large distance

    def _check_collisions(self):
        """Check for collisions with other vehicles"""
        if self.controller.use_spatial_hash:
            # Use spatial hash for efficient detection
            nearby = self.controller.spatial_hash.get_nearby_objects(self)
            for obj in nearby:
                if isinstance(obj, Vehicle) and obj is not self:
                    if self._is_colliding(obj):
                        self._handle_collision()
                        obj._handle_collision()
        else:
            # Simple collision detection against all vehicles
            for direction in self.controller.vehicles:
                if isinstance(direction, str) and direction != "crossed":
                    for lane in self.controller.vehicles[direction]:
                        if isinstance(lane, int):  # Skip non-lane entries
                            for vehicle in self.controller.vehicles[direction][lane]:
                                if vehicle is not self and not vehicle.crashed:
                                    if self._is_colliding(vehicle):
                                        self._handle_collision()
                                        vehicle._handle_collision()

    def _is_colliding(self, other):
        """Check if colliding with another vehicle"""
        # Simple rectangle collision
        return (
            self.x < other.x + other.width - self.collision_tolerance
            and self.x + self.width > other.x + self.collision_tolerance
            and self.y < other.y + other.height - self.collision_tolerance
            and self.y + self.height > other.y + self.collision_tolerance
        )

    def _handle_collision(self):
        """Handle a collision event"""
        if not self.crashed:
            self.crashed = True
            self.controller.crashes += 1

            # Change color to indicate crash
            self.image.fill((0, 0, 0, 0))  # Clear
            pygame.draw.rect(
                self.image,
                (100, 100, 100),  # Gray color for crashed
                (0, 0, self.width, self.height),
                border_radius=3,
            )

            logger.warning(f"Vehicle {self.id} crashed at ({self.x}, {self.y})")

    def update(self):
        """Update method required by pygame sprite"""
        self.move()

    def draw(self, screen):
        """Draw the vehicle on the screen"""
        # Simple draw for basic version
        screen.blit(self.image, (self.x, self.y))

    def get_state_dict(self):
        """Get a dictionary representation of vehicle state (for debugging)"""
        return {
            "id": self.id,
            "type": self.vehicleClass,
            "position": (self.x, self.y),
            "speed": self.speed,
            "waiting_time": self.waiting_time,
            "crossed": self.crossed == 1,
            "crashed": self.crashed,
            "emissions": self.emissions,
        }
