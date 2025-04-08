import pygame
import sys
import time
import random
import logging
import argparse
import numpy as np
from datetime import datetime
from modules.traffic import TrafficController, VehicleGenerator, SpatialHash
from modules.vehicle import Vehicle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("simulation.log"), logging.StreamHandler()],
)

logger = logging.getLogger("Simulation")


class TrafficSimulation:
    """Main class for traffic simulation"""

    def __init__(self, config=None):
        """Initialize simulation with optional config"""
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Traffic Intersection Simulation")

        # Default configuration
        self.config = {
            "width": 1400,  # Screen width
            "height": 800,  # Screen height
            "fps": 60,  # Frames per second
            "signal_timings": {
                "min_green": 10,  # Minimum green time
                "max_green": 30,  # Maximum green time
                "yellow": 3,  # Yellow phase duration
            },
            "vehicle_rates": {  # Vehicles per minute for each direction
                "right": 20,
                "down": 20,
                "left": 20,
                "up": 20,
            },
            "vehicle_types": {  # Percentage of each vehicle type
                "car": 70,
                "bus": 10,
                "truck": 15,
                "bike": 5,
            },
            "lanes": 2,  # Number of lanes per direction
            "use_spatial_hash": True,  # Use spatial hash for collision detection
            "grid_size": 100,  # Grid size for spatial hash
            "performance_mode": False,  # Disable some features for performance
            "seed": None,  # Random seed
            "max_vehicles": 100,  # Maximum vehicles in simulation
            "reinforcement_learning": False,  # Use RL for traffic control
        }

        # Override defaults with provided config
        if config:
            self.config.update(config)

        # Set random seed if provided
        if self.config["seed"]:
            random.seed(self.config["seed"])
            np.random.seed(self.config["seed"])

        # Initialize screen
        self.width = self.config["width"]
        self.height = self.config["height"]
        self.screen = pygame.display.set_mode((self.width, self.height))

        # Set up clock
        self.clock = pygame.time.Clock()
        self.fps = self.config["fps"]

        # Create traffic controller
        self.controller = self._create_traffic_controller()

        # Create vehicle generator
        self.generator = self._create_vehicle_generator()

        # Simulation state
        self.running = False
        self.paused = False
        self.step = 0
        self.start_time = None
        self.stats = {
            "vehicles_generated": 0,
            "vehicles_completed": 0,
            "average_wait_time": 0,
            "total_emissions": 0,
            "crashes": 0,
            "simulation_time": 0,
        }

        # Sprite groups
        self.all_vehicles = pygame.sprite.Group()

        logger.info("Traffic simulation initialized")

    def _create_traffic_controller(self):
        """Create and configure traffic controller"""
        # Create controller
        controller = TrafficController(
            x={
                "right": [25, 55],
                "down": [750, 720],
                "left": [1375, 1345],
                "up": [650, 680],
            },
            y={
                "right": [350, 380],
                "down": [25, 55],
                "left": [450, 420],
                "up": [775, 745],
            },
            signal_timings={
                "green": self.config["signal_timings"]["min_green"],
                "yellow": self.config["signal_timings"]["yellow"],
                "red": self.config["signal_timings"]["min_green"]
                + self.config["signal_timings"]["yellow"],
            },
            num_lanes=self.config["lanes"],
        )

        # Initialize spatial hash if enabled
        if self.config["use_spatial_hash"]:
            controller.use_spatial_hash = True
            controller.spatial_hash = SpatialHash(
                self.width, self.height, self.config["grid_size"]
            )

        # Set performance mode
        controller.performance_mode = self.config["performance_mode"]

        return controller

    def _create_vehicle_generator(self):
        """Create and configure vehicle generator"""
        return VehicleGenerator(
            self.controller,
            vehicle_rates={
                dir: rate / 60.0 for dir, rate in self.config["vehicle_rates"].items()
            },
            vehicle_types=self.config["vehicle_types"],
            max_vehicles=self.config["max_vehicles"],
        )

    def run(self):
        """Run the simulation"""
        self.running = True
        self.start_time = time.time()

        try:
            while self.running:
                # Process events
                self._process_events()

                # Update
                if not self.paused:
                    self._update()

                # Draw
                self._draw()

                # Cap the frame rate
                self.clock.tick(self.fps)

            # Cleanup
            self._shutdown()

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            self._shutdown()
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            self._shutdown()

    def _process_events(self):
        """Process pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_key_event(event)

    def _handle_key_event(self, event):
        """Handle keyboard events"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_SPACE:
            self.paused = not self.paused
            logger.info(f"Simulation {'paused' if self.paused else 'resumed'}")
        elif event.key == pygame.K_s:
            # Save state
            self._save_stats()
            logger.info("Simulation stats saved")

    def _update(self):
        """Update simulation state"""
        # Increment step counter
        self.step += 1

        # Generate vehicles (based on probability)
        self.generator.update()

        # Update traffic signals
        self.controller.update_signals()

        # Update all vehicles
        for direction in self.controller.vehicles:
            if isinstance(direction, str) and direction != "crossed":
                for lane in self.controller.vehicles[direction]:
                    if isinstance(lane, int):
                        for vehicle in self.controller.vehicles[direction][lane]:
                            vehicle.move()
                            # Add to sprite group if not already there
                            if vehicle not in self.all_vehicles:
                                self.all_vehicles.add(vehicle)

        # Update stats
        self._update_stats()

    def _draw(self):
        """Draw simulation elements to screen"""
        # Clear screen
        self.screen.fill((60, 60, 60))

        # Draw road
        self._draw_road()

        # Draw vehicles
        self.all_vehicles.draw(self.screen)

        # Draw traffic signals
        self._draw_signals()

        # Draw stats
        self._draw_stats()

        # Update display
        pygame.display.flip()

    def _draw_road(self):
        """Draw road and intersection"""
        # Road color
        road_color = (40, 40, 40)

        # Draw horizontal road
        pygame.draw.rect(self.screen, road_color, (0, 350, self.width, 100))

        # Draw vertical road
        pygame.draw.rect(self.screen, road_color, (650, 0, 100, self.height))

        # Draw lane markings (horizontal)
        for x in range(0, self.width, 40):
            if x < 600 or x > 800:  # Skip intersection
                pygame.draw.line(
                    self.screen, (255, 255, 255), (x, 400), (x + 20, 400), 2
                )

        # Draw lane markings (vertical)
        for y in range(0, self.height, 40):
            if y < 300 or y > 500:  # Skip intersection
                pygame.draw.line(
                    self.screen, (255, 255, 255), (700, y), (700, y + 20), 2
                )

        # Draw intersection box
        pygame.draw.rect(self.screen, (50, 50, 50), (650, 350, 100, 100))

    def _draw_signals(self):
        """Draw traffic signals"""
        # Signal positions
        signal_positions = {
            0: (630, 330),  # Right
            1: (730, 330),  # Down
            2: (730, 450),  # Left
            3: (630, 450),  # Up
        }

        # Draw signal boxes
        for i, pos in signal_positions.items():
            # Select color based on signal state
            if i == self.controller.currentGreen and self.controller.currentYellow == 0:
                color = (0, 255, 0)  # Green
            elif (
                i == self.controller.currentGreen and self.controller.currentYellow == 1
            ):
                color = (255, 255, 0)  # Yellow
            else:
                color = (255, 0, 0)  # Red

            # Draw signal
            pygame.draw.rect(self.screen, color, (pos[0], pos[1], 20, 20))

    def _draw_stats(self):
        """Draw simulation statistics"""
        # Setup font
        font = pygame.font.SysFont("Arial", 18)

        # Stats to display
        stats_text = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Time: {int(self.stats['simulation_time'])}s",
            f"Vehicles: {len(self.all_vehicles)}",
            f"Completed: {self.stats['vehicles_completed']}",
            f"Avg Wait: {self.stats['average_wait_time']:.1f}s",
            f"Crashes: {self.stats['crashes']}",
            f"CO2: {self.stats['total_emissions']:.1f}kg",
        ]

        # Status text (paused)
        if self.paused:
            pause_surf = font.render("PAUSED", True, (255, 255, 255))
            pause_rect = pause_surf.get_rect(center=(self.width // 2, 30))
            self.screen.blit(pause_surf, pause_rect)

        # Draw stats in top-left corner
        y_offset = 10
        for text in stats_text:
            text_surf = font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surf, (10, y_offset))
            y_offset += 25

    def _update_stats(self):
        """Update simulation statistics"""
        # Update simulation time
        self.stats["simulation_time"] = time.time() - self.start_time

        # Get vehicles completed from controller
        total_completed = sum(
            self.controller.vehicles[dir]["crossed"]
            for dir in ["right", "down", "left", "up"]
        )
        self.stats["vehicles_completed"] = total_completed

        # Calculate average waiting time
        all_waiting_times = []
        for direction in ["right", "down", "left", "up"]:
            all_waiting_times.extend(self.controller.waiting_times[direction].values())

        if all_waiting_times:
            self.stats["average_wait_time"] = sum(all_waiting_times) / len(
                all_waiting_times
            )

        # Update total emissions
        self.stats["total_emissions"] = sum(self.controller.emission_counts.values())

        # Update crashes
        self.stats["crashes"] = self.controller.crashes

    def _save_stats(self):
        """Save simulation statistics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write("Traffic Simulation Statistics\n")
            f.write("-" * 30 + "\n")
            f.write(f"Time: {self.stats['simulation_time']:.2f} seconds\n")
            f.write(f"Vehicles Generated: {self.stats['vehicles_generated']}\n")
            f.write(f"Vehicles Completed: {self.stats['vehicles_completed']}\n")
            f.write(
                f"Average Wait Time: {self.stats['average_wait_time']:.2f} seconds\n"
            )
            f.write(f"Total CO2 Emissions: {self.stats['total_emissions']:.2f} kg\n")
            f.write(f"Total Crashes: {self.stats['crashes']}\n")

            # Add configuration
            f.write("\nConfiguration:\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")

    def _shutdown(self):
        """Clean up and shut down simulation"""
        # Save final stats
        self._save_stats()

        # Log stats
        logger.info(
            f"Simulation completed in {self.stats['simulation_time']:.2f} seconds"
        )
        logger.info(f"Vehicles completed: {self.stats['vehicles_completed']}")
        logger.info(f"Average wait time: {self.stats['average_wait_time']:.2f} seconds")
        logger.info(f"Total crashes: {self.stats['crashes']}")

        # Quit pygame
        pygame.quit()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Traffic Intersection Simulation")
    parser.add_argument("--seed", type=int, help="Random seed for simulation")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument(
        "--lanes", type=int, default=2, help="Number of lanes per direction"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run in performance mode"
    )
    args = parser.parse_args()

    # Create config from arguments
    config = {
        "seed": args.seed,
        "fps": args.fps,
        "lanes": args.lanes,
        "performance_mode": args.performance,
    }

    # Create and run simulation
    sim = TrafficSimulation(config)
    sim.run()
