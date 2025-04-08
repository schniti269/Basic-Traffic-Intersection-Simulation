import random
import time
import threading
import logging
import numpy as np
from collections import deque
import pygame
from modules.vehicle import Vehicle, VEHICLE_SIZES
from enum import Enum
import math

logger = logging.getLogger("TrafficControl")

# Default values of signal timers
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 5

# Traffic control settings
MAX_WAITING_TIME = 100  # Max time a car can wait before maximum penalty

# CO2 emission factors for different vehicle types (g/km)
CO2_EMISSIONS = {
    "car": 120,
    "bus": 900,
    "truck": 800,
    "bike": 50,
}


# Spatial hash grid for efficient collision detection
class SpatialHash:
    def __init__(self, width, height, cell_size):
        """Initialize the spatial hash grid"""
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Calculate grid dimensions
        self.cols = width // cell_size + 1
        self.rows = height // cell_size + 1

        # Create the grid
        self.grid = {}

        logger.info(f"Spatial hash initialized: {self.cols}x{self.rows} cells")

    def _get_cell_key(self, x, y):
        """Get the key for a cell at the given coordinates"""
        col = max(0, min(int(x / self.cell_size), self.cols - 1))
        row = max(0, min(int(y / self.cell_size), self.rows - 1))
        return (col, row)

    def _get_cell_keys_for_entity(self, entity):
        """Get all cell keys that an entity occupies"""
        # Get entity bounds
        x, y = entity.rect.x, entity.rect.y
        w, h = entity.rect.width, entity.rect.height

        # Calculate the cells the entity spans
        min_col = max(0, int(x / self.cell_size))
        min_row = max(0, int(y / self.cell_size))
        max_col = min(int((x + w) / self.cell_size), self.cols - 1)
        max_row = min(int((y + h) / self.cell_size), self.rows - 1)

        # Generate all cell keys
        return [
            (col, row)
            for col in range(min_col, max_col + 1)
            for row in range(min_row, max_row + 1)
        ]

    def insert(self, entity):
        """Insert an entity into the spatial hash"""
        for cell_key in self._get_cell_keys_for_entity(entity):
            if cell_key not in self.grid:
                self.grid[cell_key] = []

            # Only add if not already there
            if entity not in self.grid[cell_key]:
                self.grid[cell_key].append(entity)

    def update(self, entity):
        """Update an entity's position in the spatial hash"""
        # Remove from old cells
        self.remove(entity)
        # Add to new cells
        self.insert(entity)

    def remove(self, entity):
        """Remove an entity from the spatial hash"""
        # Try to get cached cell keys if available
        for key, entities in list(self.grid.items()):
            if entity in entities:
                entities.remove(entity)

                # Remove empty cells
                if not entities:
                    del self.grid[key]

    def get_nearby_entities(self, entity):
        """Get all entities in the same cells as the given entity"""
        nearby = set()

        for cell_key in self._get_cell_keys_for_entity(entity):
            if cell_key in self.grid:
                for other in self.grid[cell_key]:
                    if other != entity:
                        nearby.add(other)

        return list(nearby)

    def query(self, x, y, width, height):
        """Query for all entities in a rectangular region"""
        min_col = max(0, int(x / self.cell_size))
        min_row = max(0, int(y / self.cell_size))
        max_col = min(int((x + width) / self.cell_size), self.cols - 1)
        max_row = min(int((y + height) / self.cell_size), self.rows - 1)

        # Create rect for collision checks
        rect = pygame.Rect(x, y, width, height)

        # Collect entities from all cells in the region
        entities = set()
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                cell_key = (col, row)
                if cell_key in self.grid:
                    for entity in self.grid[cell_key]:
                        # Only add if actually colliding
                        if rect.colliderect(entity.rect):
                            entities.add(entity)

        return list(entities)

    def clear(self):
        """Clear all entities from the spatial hash"""
        self.grid.clear()


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class Vehicle:
    """Class representing a vehicle in the simulation"""

    def __init__(self, vehicle_id, x, y, direction, size=(40, 20), speed=1, color=None):
        """Initialize a vehicle"""
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.direction = direction  # "up", "down", "left", "right"
        self.size = size
        self.speed = speed
        self.waiting = False
        self.wait_time = 0
        self.emissions = 0
        self.total_distance = 0

        # Set default color if none provided
        if color is None:
            # Generate a random color but avoid very dark colors
            self.color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255),
            )
        else:
            self.color = color

    def update(self, traffic_controller, intersection):
        """Update vehicle position and state"""
        # Store previous position to calculate distance traveled
        prev_x, prev_y = self.x, self.y

        # Check if vehicle is at or approaching the intersection
        is_at_intersection = self._is_at_intersection(intersection)

        # Check if traffic light is red for this direction
        if (
            is_at_intersection
            and traffic_controller.get_light_state(self.direction)
            != TrafficLightState.GREEN.value
        ):
            self.waiting = True
            self.wait_time += 1
            # Idle emissions
            self.emissions += 0.1
        else:
            # Move the vehicle according to its direction
            self._move()
            self.waiting = False

        # Calculate distance traveled in this step
        distance = math.sqrt((self.x - prev_x) ** 2 + (self.y - prev_y) ** 2)
        self.total_distance += distance

        # Calculate emissions based on speed and acceleration
        if not self.waiting:
            self.emissions += 0.05 * self.speed

    def _move(self):
        """Move the vehicle according to its direction and speed"""
        if self.direction == "right":
            self.x += self.speed
        elif self.direction == "left":
            self.x -= self.speed
        elif self.direction == "down":
            self.y += self.speed
        elif self.direction == "up":
            self.y -= self.speed

    def _is_at_intersection(self, intersection):
        """Check if the vehicle is at or approaching the intersection"""
        # Calculate the center of the vehicle
        vehicle_center_x = self.x + self.size[0] // 2
        vehicle_center_y = self.y + self.size[1] // 2

        # Check based on direction
        if self.direction == "right" and (
            vehicle_center_x >= intersection["x"] - 10
            and vehicle_center_x <= intersection["x"]
            and vehicle_center_y >= intersection["y"]
            and vehicle_center_y <= intersection["y"] + intersection["size"]
        ):
            return True
        elif self.direction == "left" and (
            vehicle_center_x <= intersection["x"] + intersection["size"] + 10
            and vehicle_center_x >= intersection["x"] + intersection["size"]
            and vehicle_center_y >= intersection["y"]
            and vehicle_center_y <= intersection["y"] + intersection["size"]
        ):
            return True
        elif self.direction == "down" and (
            vehicle_center_y >= intersection["y"] - 10
            and vehicle_center_y <= intersection["y"]
            and vehicle_center_x >= intersection["x"]
            and vehicle_center_x <= intersection["x"] + intersection["size"]
        ):
            return True
        elif self.direction == "up" and (
            vehicle_center_y <= intersection["y"] + intersection["size"] + 10
            and vehicle_center_y >= intersection["y"] + intersection["size"]
            and vehicle_center_x >= intersection["x"]
            and vehicle_center_x <= intersection["x"] + intersection["size"]
        ):
            return True

        return False

    def is_out_of_bounds(self, screen_width, screen_height):
        """Check if the vehicle is out of the screen bounds"""
        if (
            self.x < -self.size[0]
            or self.x > screen_width
            or self.y < -self.size[1]
            or self.y > screen_height
        ):
            return True
        return False

    def check_collision(self, other_vehicle):
        """Check if this vehicle collides with another vehicle"""
        # Simple rectangular collision detection
        return (
            self.x < other_vehicle.x + other_vehicle.size[0]
            and self.x + self.size[0] > other_vehicle.x
            and self.y < other_vehicle.y + other_vehicle.size[1]
            and self.y + self.size[1] > other_vehicle.y
        )


class TrafficController:
    """Class for controlling traffic flow and lights"""

    def __init__(self, config=None):
        """Initialize the traffic controller"""
        # Default configuration
        self.config = {
            "cycle_duration": 300,  # Duration of a full traffic light cycle in steps
            "yellow_duration": 30,  # Duration of yellow light in steps
            "vehicle_spawn_rate": 0.05,  # Probability of spawning a vehicle each step
            "min_vehicle_distance": 60,  # Minimum distance between vehicles
            "max_vehicles": 50,  # Maximum number of vehicles in the simulation
        }

        # Override with custom config if provided
        if config:
            self.config.update(config)

        # Traffic light states
        self.lights = {
            "right": TrafficLightState.RED.value,
            "left": TrafficLightState.RED.value,
            "up": TrafficLightState.GREEN.value,
            "down": TrafficLightState.GREEN.value,
        }

        # Active directions (which directions have green lights)
        self.active_directions = ["up", "down"]

        # Counter for the current cycle
        self.cycle_step = 0

        # Vehicle management
        self.vehicles = []
        self.next_vehicle_id = 1
        self.vehicles_processed = 0
        self.vehicles_generated = 0
        self.crashes = 0

    def update(self, intersection, screen_width, screen_height):
        """Update traffic lights and vehicles"""
        # Update traffic light cycle
        self._update_traffic_lights()

        # Spawn new vehicles with some probability
        self._spawn_vehicles(screen_width, screen_height)

        # Update each vehicle
        self._update_vehicles(intersection, screen_width, screen_height)

        # Check for collisions
        self._check_collisions()

    def _update_traffic_lights(self):
        """Update traffic light states based on the cycle"""
        self.cycle_step = (self.cycle_step + 1) % self.config["cycle_duration"]

        half_cycle = self.config["cycle_duration"] // 2
        yellow_duration = self.config["yellow_duration"]

        # First half of cycle: north-south green, east-west red
        if self.cycle_step < half_cycle:
            if self.cycle_step < half_cycle - yellow_duration:
                self.lights["up"] = TrafficLightState.GREEN.value
                self.lights["down"] = TrafficLightState.GREEN.value
                self.lights["left"] = TrafficLightState.RED.value
                self.lights["right"] = TrafficLightState.RED.value
                self.active_directions = ["up", "down"]
            else:
                # Yellow transition
                self.lights["up"] = TrafficLightState.YELLOW.value
                self.lights["down"] = TrafficLightState.YELLOW.value
                self.lights["left"] = TrafficLightState.RED.value
                self.lights["right"] = TrafficLightState.RED.value
                self.active_directions = []
        # Second half: north-south red, east-west green
        else:
            if self.cycle_step < self.config["cycle_duration"] - yellow_duration:
                self.lights["up"] = TrafficLightState.RED.value
                self.lights["down"] = TrafficLightState.RED.value
                self.lights["left"] = TrafficLightState.GREEN.value
                self.lights["right"] = TrafficLightState.GREEN.value
                self.active_directions = ["left", "right"]
            else:
                # Yellow transition
                self.lights["up"] = TrafficLightState.RED.value
                self.lights["down"] = TrafficLightState.RED.value
                self.lights["left"] = TrafficLightState.YELLOW.value
                self.lights["right"] = TrafficLightState.YELLOW.value
                self.active_directions = []

    def _spawn_vehicles(self, screen_width, screen_height):
        """Spawn new vehicles at the edges of the screen"""
        # Limit the number of vehicles
        if len(self.vehicles) >= self.config["max_vehicles"]:
            return

        # Probability check for spawning
        if random.random() > self.config["vehicle_spawn_rate"]:
            return

        # Select a random direction
        direction = random.choice(["up", "down", "left", "right"])

        # Calculate spawn position based on direction
        if direction == "right":
            x = 0
            y = screen_height // 2 + random.randint(-20, 20)
        elif direction == "left":
            x = screen_width
            y = screen_height // 2 + random.randint(-20, 20)
        elif direction == "down":
            x = screen_width // 2 + random.randint(-20, 20)
            y = 0
        elif direction == "up":
            x = screen_width // 2 + random.randint(-20, 20)
            y = screen_height

        # Check if there's enough space
        if not self._is_space_available(x, y, direction):
            return

        # Create and add the new vehicle
        vehicle = Vehicle(
            self.next_vehicle_id, x, y, direction, speed=random.uniform(0.5, 2.0)
        )
        self.vehicles.append(vehicle)
        self.next_vehicle_id += 1
        self.vehicles_generated += 1

    def _is_space_available(self, x, y, direction):
        """Check if there's enough space to spawn a vehicle"""
        min_distance = self.config["min_vehicle_distance"]

        for vehicle in self.vehicles:
            if vehicle.direction != direction:
                continue

            distance = 0
            if direction == "right":
                distance = abs(vehicle.x - x)
            elif direction == "left":
                distance = abs(vehicle.x - x)
            elif direction == "down":
                distance = abs(vehicle.y - y)
            elif direction == "up":
                distance = abs(vehicle.y - y)

            if distance < min_distance:
                return False

        return True

    def _update_vehicles(self, intersection, screen_width, screen_height):
        """Update all vehicles and remove those that are out of bounds"""
        vehicles_to_remove = []

        for vehicle in self.vehicles:
            vehicle.update(self, intersection)

            # Check if vehicle is out of bounds
            if vehicle.is_out_of_bounds(screen_width, screen_height):
                vehicles_to_remove.append(vehicle)

        # Remove vehicles that are out of bounds
        for vehicle in vehicles_to_remove:
            self.vehicles.remove(vehicle)
            self.vehicles_processed += 1

    def _check_collisions(self):
        """Check for collisions between vehicles"""
        # Simple O(n²) collision detection
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                if self.vehicles[i].check_collision(self.vehicles[j]):
                    # Log collision
                    logger.warning(
                        f"Collision detected between vehicles {self.vehicles[i].id} and {self.vehicles[j].id}"
                    )
                    self.crashes += 1

                    # Optional: remove crashed vehicles
                    # self.vehicles.remove(self.vehicles[j])
                    # self.vehicles.remove(self.vehicles[i])
                    # Break out of loop since we modified the array
                    # break

    def get_all_vehicles(self):
        """Return all vehicles"""
        return self.vehicles

    def get_waiting_vehicles(self):
        """Return vehicles that are waiting at traffic lights"""
        return [v for v in self.vehicles if v.waiting]

    def get_traffic_lights(self):
        """Return the current state of all traffic lights"""
        return self.lights

    def get_light_state(self, direction):
        """Get the state of a specific traffic light"""
        return self.lights.get(direction, TrafficLightState.RED.value)

    def get_stats(self):
        """Get traffic statistics"""
        total_wait_time = sum(v.wait_time for v in self.vehicles)
        total_emissions = sum(v.emissions for v in self.vehicles)

        # Calculate average wait time
        if self.vehicles_processed > 0:
            avg_wait_time = total_wait_time / max(1, len(self.vehicles))
        else:
            avg_wait_time = 0

        return {
            "vehicles_generated": self.vehicles_generated,
            "vehicles_processed": self.vehicles_processed,
            "active_vehicles": len(self.vehicles),
            "waiting_vehicles": len(self.get_waiting_vehicles()),
            "average_wait_time": avg_wait_time,
            "total_emissions": total_emissions,
            "crashes": self.crashes,
        }

    def reset(self):
        """Reset the traffic controller"""
        self.vehicles = []
        self.cycle_step = 0
        self.vehicles_processed = 0
        self.vehicles_generated = 0
        self.crashes = 0


class TrafficController:
    """Manages traffic signals and vehicle movement logic"""

    def __init__(self, performance_mode=False):
        self.signals = []
        self.noOfSignals = 4
        self.currentGreen = 0  # Indicates which signal is green currently
        self.nextGreen = 1  # Indicates which signal will turn green next
        self.currentYellow = 0  # Indicates whether yellow signal is on or off

        # Speed settings for different vehicle types
        self.speeds = {
            "car": 2.25,
            "bus": 1.8,
            "truck": 1.8,
            "bike": 2.5,
        }

        # Coordinates mapping
        self.directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}
        self.vehicleTypes = {0: "car", 1: "bus", 2: "truck", 3: "bike"}

        # Gap between vehicles
        self.stoppingGap = 15  # stopping gap
        self.movingGap = 15  # moving gap

        # Coordinates of stop lines
        self.stopLines = {"right": 590, "down": 330, "left": 800, "up": 535}
        self.defaultStop = {"right": 580, "down": 320, "left": 810, "up": 545}

        # Coordinates of vehicles' start
        self.x = {
            "right": [0, 0, 0],
            "down": [755, 727, 697],
            "left": [1400, 1400, 1400],
            "up": [602, 627, 657],
        }
        self.y = {
            "right": [348, 370, 398],
            "down": [0, 0, 0],
            "left": [498, 466, 436],
            "up": [800, 800, 800],
        }

        # Vehicle tracking
        self.vehicles = {
            "right": {0: [], 1: [], 2: [], "crossed": 0},
            "down": {0: [], 1: [], 2: [], "crossed": 0},
            "left": {0: [], 1: [], 2: [], "crossed": 0},
            "up": {0: [], 1: [], 2: [], "crossed": 0},
        }

        # Metrics for RL
        self.waiting_vehicles = {"right": 0, "down": 0, "left": 0, "up": 0}
        self.waiting_by_type = {"car": 0, "bus": 0, "truck": 0, "bike": 0}
        self.emission_counts = {"right": 0, "down": 0, "left": 0, "up": 0}
        self.crashes = 0
        self.total_reward = 0
        self.waiting_times = {"right": {}, "down": {}, "left": {}, "up": {}}

        # Performance mode settings
        self.performance_mode = performance_mode
        self.spatial_hash = SpatialHash(50, 50, 50)  # 50px cell size
        self.use_spatial_hash = True

        # Initialize signals
        self.initialize_signals()

        # Pygame simulation group (will be populated from outside)
        self.simulation = None

        logger.info("TrafficController initialized")

    def initialize_signals(self):
        """Initialize traffic signals with default timings"""
        self.signals = []
        ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
        self.signals.append(ts1)
        ts2 = TrafficSignal(
            ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen[1]
        )
        self.signals.append(ts2)
        ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
        self.signals.append(ts3)
        ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
        self.signals.append(ts4)
        logger.info("Traffic signals initialized")

    def set_simulation_group(self, simulation_group):
        """Set the pygame sprite group for simulation"""
        self.simulation = simulation_group

    def update_signal_values(self):
        """Update values of the signal timers after every second"""
        for i in range(0, self.noOfSignals):
            if i == self.currentGreen:
                if self.currentYellow == 0:
                    self.signals[i].green -= 1
                else:
                    self.signals[i].yellow -= 1
            else:
                self.signals[i].red -= 1

    def set_next_green(self, signal_index):
        """Set which signal will turn green next"""
        if 0 <= signal_index < self.noOfSignals:
            self.nextGreen = signal_index
            logger.debug(f"Next green signal set to {signal_index}")
            return True
        return False

    def process_yellow_phase(self):
        """Process the yellow signal phase"""
        self.currentYellow = 1  # set yellow signal on
        # reset stop coordinates of lanes and vehicles
        for i in range(0, 3):
            for vehicle in self.vehicles[self.directionNumbers[self.currentGreen]][i]:
                vehicle.stop = self.defaultStop[
                    self.directionNumbers[self.currentGreen]
                ]

        # Wait for yellow phase to complete
        while self.signals[self.currentGreen].yellow > 0:
            self.update_signal_values()
            time.sleep(1)

        self.currentYellow = 0  # set yellow signal off

        # Reset all signal times of current signal to default times
        self.signals[self.currentGreen].green = defaultGreen[self.currentGreen]
        self.signals[self.currentGreen].yellow = defaultYellow
        self.signals[self.currentGreen].red = defaultRed

        # Update current and next green signals
        self.currentGreen = self.nextGreen
        self.nextGreen = (self.currentGreen + 1) % self.noOfSignals

        # Set the red time of next signal
        self.signals[self.nextGreen].red = (
            self.signals[self.currentGreen].yellow
            + self.signals[self.currentGreen].green
        )

    def get_waiting_vehicles_count(self):
        """Get count of waiting vehicles by direction and type"""
        # Reset counts
        self.waiting_vehicles = {"right": 0, "down": 0, "left": 0, "up": 0}
        self.waiting_by_type = {"car": 0, "bus": 0, "truck": 0, "bike": 0}

        # Count vehicles that are waiting
        for direction in self.directionNumbers.values():
            for lane in range(3):
                for vehicle in self.vehicles[direction][lane]:
                    if vehicle.waiting_time > 0:
                        self.waiting_vehicles[direction] += 1
                        self.waiting_by_type[vehicle.vehicleClass] += 1

        return {"by_direction": self.waiting_vehicles, "by_type": self.waiting_by_type}

    def get_state(self):
        """Get the current state of the traffic system for the RL agent"""
        # Count waiting vehicles at each signal
        waiting_count = [0, 0, 0, 0]
        for direction_idx, direction in self.directionNumbers.items():
            waiting = 0
            for lane in range(3):
                for vehicle in self.vehicles[direction][lane]:
                    if vehicle.waiting_time > 0:
                        waiting += 1
            waiting_count[direction_idx] = waiting

        # Get current signal state
        signal_state = [0] * 4
        signal_state[self.currentGreen] = 1 if self.currentYellow == 0 else 0.5

        # Get queue length in each direction
        queue_lengths = []
        for direction in self.directionNumbers.values():
            total = 0
            for lane in range(3):
                total += len(self.vehicles[direction][lane])
            queue_lengths.append(total)

        # Create state vector
        state = waiting_count + signal_state + queue_lengths + [self.crashes]
        return np.array(state)

    def get_vehicle_type_state(self):
        """Get state with additional vehicle type information"""
        base_state = self.get_state()

        # Count waiting vehicles by type
        type_counts = []
        for vtype in self.vehicleTypes.values():
            for direction in self.directionNumbers.values():
                type_count = 0
                for lane in range(3):
                    for vehicle in self.vehicles[direction][lane]:
                        if vehicle.vehicleClass == vtype and vehicle.waiting_time > 0:
                            type_count += 1
                type_counts.append(type_count)

        # Combine base state with type information
        return np.concatenate([base_state, np.array(type_counts)])

    def get_metrics(self):
        """Get current traffic metrics for reward calculation"""
        return {
            "waiting_vehicles": self.waiting_by_type,
            "emission_counts": self.emission_counts,
            "crashes": self.crashes,
            "waiting_times": self.waiting_times,
        }

    def add_vehicle(self, lane, vehicle_class, direction_number):
        """Add a new vehicle to the simulation (called from VehicleGenerator)"""
        if self.simulation is None:
            logger.error("Cannot add vehicle: simulation group not set")
            return None

        direction = self.directionNumbers[direction_number]

        # Create a new Vehicle and add it to the collections
        from modules.vehicle import Vehicle

        try:
            vehicle = Vehicle(
                lane=lane,
                vehicleClass=vehicle_class,
                direction_number=direction_number,
                direction=direction,
                controller=self,
            )
            self.simulation.add(vehicle)
            return vehicle
        except Exception as e:
            logger.error(f"Error creating vehicle: {e}")
            return None

    def reset(self):
        """Reset the traffic controller state"""
        # Reset all environment variables
        self.vehicles = {
            "right": {0: [], 1: [], 2: [], "crossed": 0},
            "down": {0: [], 1: [], 2: [], "crossed": 0},
            "left": {0: [], 1: [], 2: [], "crossed": 0},
            "up": {0: [], 1: [], 2: [], "crossed": 0},
        }
        self.waiting_vehicles = {"right": 0, "down": 0, "left": 0, "up": 0}
        self.waiting_by_type = {"car": 0, "bus": 0, "truck": 0, "bike": 0}
        self.emission_counts = {"right": 0, "down": 0, "left": 0, "up": 0}
        self.waiting_times = {"right": {}, "down": {}, "left": {}, "up": {}}
        self.crashes = 0
        self.total_reward = 0

        self.currentGreen = 0
        self.currentYellow = 0
        self.nextGreen = 1

        # Re-initialize signals
        self.initialize_signals()

        # Clear spatial hash
        self.spatial_hash.clear()

        logger.info("TrafficController reset")

        # Return initial state
        return self.get_state()

    def get_green_time_left(self):
        """Get remaining green time for current signal"""
        if self.currentYellow == 0:
            return self.signals[self.currentGreen].green
        return 0

    def get_total_emission(self):
        """Get total emissions from all directions"""
        return sum(self.emission_counts.values())

    def get_total_waiting(self):
        """Get total waiting vehicles"""
        return sum(self.waiting_vehicles.values())

    def get_statistics(self):
        """Get comprehensive traffic statistics"""
        total_vehicles = 0
        crossed_vehicles = 0
        waiting_count = 0

        for direction in self.directionNumbers.values():
            crossed_vehicles += self.vehicles[direction]["crossed"]
            for lane in range(3):
                total_vehicles += len(self.vehicles[direction][lane])
                for vehicle in self.vehicles[direction][lane]:
                    if vehicle.waiting_time > 0:
                        waiting_count += 1

        return {
            "total_vehicles": total_vehicles,
            "crossed_vehicles": crossed_vehicles,
            "waiting_vehicles": waiting_count,
            "crashes": self.crashes,
            "total_emissions": self.get_total_emission(),
            "green_signal": self.currentGreen,
            "yellow_active": self.currentYellow == 1,
        }


class VehicleGenerator:
    """Generates vehicles based on traffic patterns"""

    def __init__(self, traffic_controller, performance_mode=False):
        self.traffic_controller = traffic_controller
        self.performance_mode = performance_mode
        self.MAX_VEHICLES_PER_SECOND = 10  # Adjust based on performance
        self.vehicle_count = 0
        self.last_reset_time = time.time()

        # Traffic pattern parameters - Rush hour simulation with varying density
        self.RUSH_HOUR_DENSITY = {
            "morning": {
                "car": 70,
                "bus": 10,
                "truck": 15,
                "bike": 5,
            },  # Morning rush hour
            "midday": {"car": 40, "bus": 5, "truck": 30, "bike": 25},  # Midday
            "evening": {
                "car": 75,
                "bus": 15,
                "truck": 5,
                "bike": 5,
            },  # Evening rush hour
            "night": {"car": 60, "bus": 5, "truck": 30, "bike": 5},  # Night time
        }

        # Time-based distribution (cycle through traffic patterns)
        self.current_pattern = "morning"
        self.pattern_change_time = (
            time.time() + 60
        )  # Change pattern every minute for testing

        # Direction probability distribution (can be dynamic)
        self.direction_dist = [25, 25, 25, 25]  # Equal distribution initially

        logger.info("VehicleGenerator initialized")

    def start(self):
        """Start the vehicle generation thread"""
        self.generator_thread = threading.Thread(
            name="generateVehicles", target=self._generate_vehicles_loop, daemon=True
        )
        self.generator_thread.start()
        logger.info("Vehicle generation started")

    def _generate_vehicles_loop(self):
        """Main loop for vehicle generation"""
        while True:
            try:
                # Rate limiting check
                current_time = time.time()
                if current_time - self.last_reset_time >= 1.0:
                    self.vehicle_count = 0
                    self.last_reset_time = current_time

                # Update traffic pattern if needed
                if current_time >= self.pattern_change_time:
                    self._update_traffic_pattern()

                # Check if we've reached the rate limit
                if self.vehicle_count >= self.MAX_VEHICLES_PER_SECOND:
                    time.sleep(0.05)  # Small delay to prevent CPU hogging
                    continue

                # Generate a new vehicle
                self._create_vehicle()

                # Shorter delay for more vehicles
                time.sleep(random.uniform(0.05, 0.15))  # Variable delay

            except Exception as e:
                logger.error(f"Error in vehicle generation: {e}")
                time.sleep(0.5)  # Delay to prevent error spam

    def _update_traffic_pattern(self):
        """Update traffic pattern based on time"""
        # Cycle through patterns
        if self.current_pattern == "morning":
            self.current_pattern = "midday"
        elif self.current_pattern == "midday":
            self.current_pattern = "evening"
        elif self.current_pattern == "evening":
            self.current_pattern = "night"
        else:
            self.current_pattern = "morning"

        # Update direction distribution based on pattern
        if self.current_pattern == "morning":
            self.direction_dist = [40, 20, 30, 10]  # More right and left traffic
        elif self.current_pattern == "evening":
            self.direction_dist = [30, 10, 40, 20]  # More left and right traffic
        else:
            self.direction_dist = [25, 25, 25, 25]  # Equal distribution

        self.pattern_change_time = (
            time.time() + 60
        )  # Next change in 1 minute for testing
        logger.info(f"Traffic pattern changed to {self.current_pattern}")

    def _create_vehicle(self):
        """Create a new vehicle based on current traffic pattern"""
        try:
            # Determine vehicle type based on pattern density
            vehicle_probs = list(self.RUSH_HOUR_DENSITY[self.current_pattern].values())
            total_prob = sum(vehicle_probs)

            # Normalize probabilities
            vehicle_probs = [p / total_prob * 100 for p in vehicle_probs]

            # Generate vehicle based on probabilities
            rand_val = random.random() * 100
            cumulative = 0

            # Map the random value to a vehicle type
            for i, prob in enumerate(vehicle_probs):
                cumulative += prob
                if rand_val <= cumulative:
                    vehicle_type = i
                    break
            else:
                vehicle_type = 0  # Default to car if something goes wrong

            # Get the vehicle class name
            vehicle_class = self.traffic_controller.vehicleTypes[vehicle_type]

            # Determine lane (0-2)
            lane_number = random.randint(0, 2)

            # Determine direction using the probability distribution
            temp = random.randint(0, 99)
            cum_dist = [self.direction_dist[0]]
            for i in range(1, len(self.direction_dist)):
                cum_dist.append(cum_dist[i - 1] + self.direction_dist[i])

            # Find direction index
            for i, threshold in enumerate(cum_dist):
                if temp < threshold:
                    direction_number = i
                    break
            else:
                direction_number = 0  # Default to right if something goes wrong

            # Create the vehicle with validation
            if (
                0 <= vehicle_type < 4
                and 0 <= lane_number <= 2
                and 0 <= direction_number < 4
            ):
                # Add vehicle to traffic controller
                vehicle = self.traffic_controller.add_vehicle(
                    lane_number, vehicle_class, direction_number
                )

                if vehicle:
                    self.vehicle_count += 1
                    logger.debug(
                        f"Vehicle created: {vehicle_class} in {self.traffic_controller.directionNumbers[direction_number]} direction, lane {lane_number}"
                    )
            else:
                logger.warning(
                    f"Invalid vehicle parameters: type={vehicle_type}, lane={lane_number}, direction={direction_number}"
                )
        except Exception as e:
            logger.error(f"Error creating vehicle: {e}", exc_info=True)

    def set_generation_rate(self, vehicles_per_second):
        """Set the vehicle generation rate"""
        if 1 <= vehicles_per_second <= 30:
            self.MAX_VEHICLES_PER_SECOND = vehicles_per_second
            logger.info(
                f"Vehicle generation rate set to {vehicles_per_second} per second"
            )
            return True
        return False
