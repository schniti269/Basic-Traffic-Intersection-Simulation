import pygame
import logging
from datetime import datetime

logger = logging.getLogger("Rendering")


class Renderer:
    """Class responsible for rendering the simulation"""

    def __init__(self, screen, traffic_controller):
        """Initialize the renderer"""
        self.screen = screen
        self.traffic_controller = traffic_controller
        self.width = screen.get_width()
        self.height = screen.get_height()

        # Road and intersection dimensions
        self.road_width = 60
        self.intersection_size = self.road_width * 2
        self.intersection_x = (self.width - self.intersection_size) // 2
        self.intersection_y = (self.height - self.intersection_size) // 2

        # Colors
        self.colors = {
            "background": (100, 100, 100),
            "road": (50, 50, 50),
            "road_lines": (255, 255, 255),
            "intersection": (80, 80, 80),
            "text": (255, 255, 255),
            "red": (255, 0, 0),
            "yellow": (255, 255, 0),
            "green": (0, 255, 0),
        }

        # Fonts
        pygame.font.init()
        self.fonts = {
            "small": pygame.font.SysFont("Arial", 14),
            "medium": pygame.font.SysFont("Arial", 20),
            "large": pygame.font.SysFont("Arial", 24),
        }

    def render(self, stats, current_step, performance_mode=False):
        """Render the current state of the simulation"""
        # Clear the screen
        self.screen.fill(self.colors["background"])

        # Draw the environment
        self._draw_environment()

        # Draw traffic lights
        self._draw_traffic_lights()

        # Draw vehicles
        self._draw_vehicles()

        # Draw statistics
        if not performance_mode:
            self._draw_stats(stats, current_step)

        # Update the display
        pygame.display.flip()

    def _draw_environment(self):
        """Draw the road network and intersection"""
        # Draw horizontal road
        horizontal_road = pygame.Rect(
            0, self.intersection_y, self.width, self.intersection_size
        )
        pygame.draw.rect(self.screen, self.colors["road"], horizontal_road)

        # Draw vertical road
        vertical_road = pygame.Rect(
            self.intersection_x, 0, self.intersection_size, self.height
        )
        pygame.draw.rect(self.screen, self.colors["road"], vertical_road)

        # Draw intersection
        intersection = pygame.Rect(
            self.intersection_x,
            self.intersection_y,
            self.intersection_size,
            self.intersection_size,
        )
        pygame.draw.rect(self.screen, self.colors["intersection"], intersection)

        # Draw lane markings - horizontal
        lane_width = self.intersection_size // 4
        for i in range(1, 4):
            if i == 2:  # Center line
                line_type = 0  # Solid
            else:
                line_type = 1  # Dashed

            y = self.intersection_y + i * lane_width
            self._draw_line(0, y, self.width, y, line_type)

        # Draw lane markings - vertical
        for i in range(1, 4):
            if i == 2:  # Center line
                line_type = 0  # Solid
            else:
                line_type = 1  # Dashed

            x = self.intersection_x + i * lane_width
            self._draw_line(x, 0, x, self.height, line_type)

    def _draw_line(self, x1, y1, x2, y2, line_type=0):
        """Draw a line on the road
        line_type: 0 for solid, 1 for dashed
        """
        if line_type == 0:  # Solid
            pygame.draw.line(
                self.screen, self.colors["road_lines"], (x1, y1), (x2, y2), 2
            )
        else:  # Dashed
            dash_length = 20
            gap_length = 10

            # Calculate direction and length
            dx, dy = x2 - x1, y2 - y1
            distance = max(abs(dx), abs(dy))

            # Normalize direction
            if distance > 0:
                dx, dy = dx / distance, dy / distance

            # Draw dashes
            pos = (x1, y1)
            dash = True
            distance_drawn = 0

            while distance_drawn < distance:
                start_pos = pos
                length = dash_length if dash else gap_length
                distance_to_draw = min(length, distance - distance_drawn)

                end_x = pos[0] + dx * distance_to_draw
                end_y = pos[1] + dy * distance_to_draw
                end_pos = (end_x, end_y)

                if dash:
                    pygame.draw.line(
                        self.screen, self.colors["road_lines"], start_pos, end_pos, 2
                    )

                pos = end_pos
                distance_drawn += distance_to_draw
                dash = not dash

    def _draw_traffic_lights(self):
        """Draw traffic lights at the intersection"""
        traffic_lights = self.traffic_controller.get_traffic_lights()

        # Traffic light positions
        light_positions = {
            "right": (
                self.intersection_x - 20,
                self.intersection_y + self.intersection_size // 4,
            ),
            "left": (
                self.intersection_x + self.intersection_size + 5,
                self.intersection_y + 3 * self.intersection_size // 4,
            ),
            "up": (
                self.intersection_x + self.intersection_size // 4,
                self.intersection_y + self.intersection_size + 5,
            ),
            "down": (
                self.intersection_x + 3 * self.intersection_size // 4,
                self.intersection_y - 20,
            ),
        }

        # Draw each traffic light
        for direction, state in traffic_lights.items():
            position = light_positions[direction]

            # Determine light color
            if state == "red":
                color = self.colors["red"]
            elif state == "yellow":
                color = self.colors["yellow"]
            elif state == "green":
                color = self.colors["green"]

            # Draw traffic light box
            light_rect = pygame.Rect(position[0], position[1], 15, 15)
            pygame.draw.rect(self.screen, color, light_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), light_rect, 2)  # Border

    def _draw_vehicles(self):
        """Draw all vehicles in the simulation"""
        vehicles = self.traffic_controller.get_all_vehicles()

        for vehicle in vehicles:
            # Create a surface for the vehicle with the correct size
            surface = pygame.Surface(vehicle.size, pygame.SRCALPHA)

            # Fill with vehicle color
            pygame.draw.rect(
                surface, vehicle.color, (0, 0, vehicle.size[0], vehicle.size[1])
            )

            # Add a border
            pygame.draw.rect(
                surface, (0, 0, 0), (0, 0, vehicle.size[0], vehicle.size[1]), 1
            )

            # Rotate the surface if needed
            if vehicle.direction in ["up", "down"]:
                surface = pygame.transform.rotate(surface, 90)

            # Blit to the screen at the vehicle's position
            self.screen.blit(surface, (vehicle.x, vehicle.y))

            # Draw a small indicator if vehicle is waiting
            if vehicle.waiting:
                pygame.draw.circle(
                    self.screen,
                    (255, 0, 0),
                    (
                        int(vehicle.x + vehicle.size[0] // 2),
                        int(vehicle.y + vehicle.size[1] // 2),
                    ),
                    3,
                )

    def _draw_stats(self, stats, current_step):
        """Draw simulation statistics"""
        # Format the statistics
        stats_text = [
            f"Step: {current_step}",
            f"Vehicles: {len(self.traffic_controller.get_all_vehicles())}",
            f"Generated: {stats['vehicles_generated']}",
            f"Processed: {stats['vehicles_processed']}",
            f"Crashes: {stats['crashes']}",
            f"Emissions: {stats['total_emissions']:.1f}",
            f"Avg Wait: {stats['average_wait_time']:.1f}",
            f"FPS: {int(pygame.time.Clock().get_fps())}",
        ]

        # Draw a semi-transparent background
        stats_surface = pygame.Surface((200, 160), pygame.SRCALPHA)
        stats_surface.fill((0, 0, 0, 150))
        self.screen.blit(stats_surface, (10, 10))

        # Draw text
        y_offset = 15
        for text in stats_text:
            text_surface = self.fonts["small"].render(text, True, self.colors["text"])
            self.screen.blit(text_surface, (20, y_offset))
            y_offset += 20

        # Draw current time
        time_text = datetime.now().strftime("%H:%M:%S")
        time_surface = self.fonts["small"].render(time_text, True, self.colors["text"])
        self.screen.blit(time_surface, (self.width - 80, 10))

        # Draw help text
        help_text = (
            "P: Pause | SPACE: Step | R: Reset | ESC: Quit | F: Toggle Performance Mode"
        )
        help_surface = self.fonts["small"].render(help_text, True, self.colors["text"])
        self.screen.blit(help_surface, (10, self.height - 25))
