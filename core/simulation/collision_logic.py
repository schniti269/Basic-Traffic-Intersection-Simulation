import pygame
from shared.utils import logger

COLLISION_TOLERANCE = 2  # Allow a small overlap (pixels) before declaring crash


def check_collision(vehicle1, vehicle2):
    """Checks for collision between two vehicles with improved logic."""

    # Prevent checking against self or already crashed vehicles (redundant check, but safe)
    if vehicle1 is vehicle2 or vehicle1.crashed or vehicle2.crashed:
        return False

    # 1. Broad Phase: Basic AABB check (efficiently discard non-overlapping pairs)
    # Ensure rects are updated to current positions if move logic updates x/y before calling this
    rect1 = pygame.Rect(vehicle1.x, vehicle1.y, vehicle1.width, vehicle1.height)
    rect2 = pygame.Rect(vehicle2.x, vehicle2.y, vehicle2.width, vehicle2.height)

    if not rect1.colliderect(rect2):
        return False  # No collision if bounding boxes don't overlap

    # 2. Narrow Phase: More specific checks for overlapping boxes

    # Case 1: Same direction, same lane (Check for rear-ending)
    if vehicle1.direction == vehicle2.direction and vehicle1.lane == vehicle2.lane:
        # Determine which vehicle is ahead based on direction
        leader, follower = None, None
        is_collision = False
        if vehicle1.direction == "right":
            leader = vehicle1 if vehicle1.x > vehicle2.x else vehicle2
            follower = vehicle2 if vehicle1.x > vehicle2.x else vehicle1
            # Check if follower's front bumper is past leader's rear bumper
            if follower.x + follower.width > leader.x + COLLISION_TOLERANCE:
                is_collision = True
        elif vehicle1.direction == "left":
            leader = vehicle1 if vehicle1.x < vehicle2.x else vehicle2
            follower = vehicle2 if vehicle1.x < vehicle2.x else vehicle1
            # Check if follower's front bumper is past leader's rear bumper
            if follower.x < leader.x + leader.width - COLLISION_TOLERANCE:
                is_collision = True
        elif vehicle1.direction == "down":
            leader = vehicle1 if vehicle1.y > vehicle2.y else vehicle2
            follower = vehicle2 if vehicle1.y > vehicle2.y else vehicle1
            # Check if follower's front bumper is past leader's rear bumper
            if follower.y + follower.height > leader.y + COLLISION_TOLERANCE:
                is_collision = True
        elif vehicle1.direction == "up":
            leader = vehicle1 if vehicle1.y < vehicle2.y else vehicle2
            follower = vehicle2 if vehicle1.y < vehicle2.y else vehicle1
            # Check if follower's front bumper is past leader's rear bumper
            if follower.y < leader.y + leader.height - COLLISION_TOLERANCE:
                is_collision = True

        if is_collision and leader and follower:
            logger.debug(f"Collision (Same Lane): {follower.id} rear-ended {leader.id}")
            return True
        else:
            # If same direction/lane but not overlapping front/back, it's not a collision
            return False

    # Case 2: Different directions or different lanes (Intersection collision)
    # AABB overlap is considered a collision in this case.
    logger.debug(
        f"Collision (Intersection/Cross-Lane): {vehicle1.id} and {vehicle2.id} bounding boxes overlap."
    )
    return True
