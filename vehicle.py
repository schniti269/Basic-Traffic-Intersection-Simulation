import pygame
from utils import (
    speeds,
    x,
    y,
    vehicles,
    waiting_times,
    stoppingGap,
    movingGap,
    stopLines,
    defaultStop,
    currentGreen,
    currentYellow,
    emission_counts,
    crashes,
    simulation,
)


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
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
        vehicles[direction][lane].append(self)

        # Initialize waiting time for this vehicle ID
        if self.id not in waiting_times[direction]:
            waiting_times[direction][self.id] = 0

        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)

        if (
            len(vehicles[direction][lane]) > 1
            and vehicles[direction][lane][self.index - 1].crossed == 0
        ):  # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
            if direction == "right":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].image.get_rect().width
                    - stoppingGap
                )  # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            elif direction == "left":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].image.get_rect().width
                    + stoppingGap
                )
            elif direction == "down":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    - vehicles[direction][lane][self.index - 1].image.get_rect().height
                    - stoppingGap
                )
            elif direction == "up":
                self.stop = (
                    vehicles[direction][lane][self.index - 1].stop
                    + vehicles[direction][lane][self.index - 1].image.get_rect().height
                    + stoppingGap
                )
        else:
            self.stop = defaultStop[direction]

        # Set new starting and stopping coordinate
        if direction == "right":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif direction == "left":
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif direction == "down":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif direction == "up":
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        old_x, old_y = self.x, self.y
        movement_occurred = False

        if self.direction == "right":
            if (
                self.crossed == 0
                and self.x + self.image.get_rect().width > stopLines[self.direction]
            ):  # if the image has crossed stop line now
                self.crossed = 1
            should_move = (
                self.x + self.image.get_rect().width <= self.stop
                or self.crossed == 1
                or (currentGreen == 0 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.x + self.image.get_rect().width
                < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap)
            )
            if should_move:
                self.x += self.speed  # move the vehicle
                movement_occurred = True
        elif self.direction == "down":
            if (
                self.crossed == 0
                and self.y + self.image.get_rect().height > stopLines[self.direction]
            ):
                self.crossed = 1
            should_move = (
                self.y + self.image.get_rect().height <= self.stop
                or self.crossed == 1
                or (currentGreen == 1 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.y + self.image.get_rect().height
                < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap)
            )
            if should_move:
                self.y += self.speed
                movement_occurred = True
        elif self.direction == "left":
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
            should_move = (
                self.x >= self.stop
                or self.crossed == 1
                or (currentGreen == 2 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.x
                > (
                    vehicles[self.direction][self.lane][self.index - 1].x
                    + vehicles[self.direction][self.lane][self.index - 1]
                    .image.get_rect()
                    .width
                    + movingGap
                )
            )
            if should_move:
                self.x -= self.speed
                movement_occurred = True
        elif self.direction == "up":
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
            should_move = (
                self.y >= self.stop
                or self.crossed == 1
                or (currentGreen == 3 and currentYellow == 0)
            ) and (
                self.index == 0
                or self.y
                > (
                    vehicles[self.direction][self.lane][self.index - 1].y
                    + vehicles[self.direction][self.lane][self.index - 1]
                    .image.get_rect()
                    .height
                    + movingGap
                )
            )
            if should_move:
                self.y -= self.speed
                movement_occurred = True

        # Update metrics for RL
        if not movement_occurred:
            # Vehicle is waiting
            # Ensure the vehicle ID exists in the waiting_times dictionary
            if self.id not in waiting_times[self.direction]:
                waiting_times[self.direction][self.id] = 0

            waiting_times[self.direction][self.id] += 1
            self.waiting_time += 1
            if old_x != self.x or old_y != self.y:
                # Vehicle had to stop (was moving and now stopped)
                self.decelerated = True
                emission_counts[self.direction] += 1
        elif self.waiting_time > 0 and (old_x != self.x or old_y != self.y):
            # Vehicle started moving after waiting
            self.accelerated = True
            emission_counts[self.direction] += 1
            self.waiting_time = 0

        # Check for collision (simplified version)
        if movement_occurred and not self.crashed:
            for vehicle in simulation:
                if vehicle != self and not vehicle.crashed:
                    # Simple rectangular collision detection
                    if (
                        abs(self.x - vehicle.x) < 20
                        and abs(self.y - vehicle.y) < 20
                        and self.direction != vehicle.direction
                    ):
                        global crashes
                        crashes += 1
                        self.crashed = True
                        vehicle.crashed = True
                        return


# Generating vehicles in the simulation
def generateVehicles():
    import random
    import time
    from utils import vehicleTypes, directionNumbers, logger

    # Rate limiting - maximum vehicles per second
    MAX_VEHICLES_PER_SECOND = 5
    vehicle_count = 0
    last_reset_time = time.time()

    logger.info("Vehicle generation started")

    while True:
        # Rate limiting check
        current_time = time.time()
        if current_time - last_reset_time >= 1.0:
            vehicle_count = 0
            last_reset_time = current_time

        # Check if we've reached the rate limit
        if vehicle_count >= MAX_VEHICLES_PER_SECOND:
            time.sleep(0.1)  # Small delay to prevent CPU hogging
            continue

        # Input validation for vehicle parameters
        vehicle_type = random.randint(0, 3)
        lane_number = random.randint(1, 2)
        temp = random.randint(0, 99)
        direction_number = 0
        dist = [25, 50, 75, 100]

        # Validate direction number
        if temp < dist[0]:
            direction_number = 0
        elif temp < dist[1]:
            direction_number = 1
        elif temp < dist[2]:
            direction_number = 2
        elif temp < dist[3]:
            direction_number = 3
        else:
            # This should never happen, but just in case
            direction_number = 0

        # Validate vehicle type and lane number
        if vehicle_type not in vehicleTypes or lane_number not in [1, 2]:
            logger.warning(
                f"Invalid vehicle parameters: type={vehicle_type}, lane={lane_number}"
            )
            continue

        # Create the vehicle
        try:
            Vehicle(
                lane_number,
                vehicleTypes[vehicle_type],
                direction_number,
                directionNumbers[direction_number],
            )
            vehicle_count += 1
            logger.debug(
                f"Vehicle created: {vehicleTypes[vehicle_type]} in {directionNumbers[direction_number]} direction, lane {lane_number}"
            )
        except Exception as e:
            logger.error(f"Error creating vehicle: {e}")

        time.sleep(1)
