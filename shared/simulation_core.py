"""
Zentrale Simulationslogik für die Verkehrssimulation.
Diese Datei enthält gemeinsame Funktionen, die sowohl vom Training (train.py)
als auch von der visuellen Ausführung (run_model.py) verwendet werden.
"""

import sys
import os
import numpy as np
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.utils import (
    logger,
    directionNumbers,
    waiting_times,
    noOfSignals,
    simulation,
    vehicles,
    stopLines,
    movingGap,
    defaultStop,
    PERFORMANCE_MODE,
    crashes
)
from core.simulation.traffic_signal import initialize
from core.simulation.vehicle import generateVehicles

# Standardkonstanten
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800

def reset_simulation():
    """Setzt die Simulationsumgebung auf einen sauberen Ausgangszustand zurück."""
    logger.info("Resetting simulation environment...")
    global simulation, vehicles, waiting_times, crashes

    # Clear vehicle collections
    simulation.empty()  # Clear the Pygame sprite group
    
    # Reset vehicle dictionaries
    for direction in vehicles:
        vehicles[direction] = {
            0: [],
            1: [],
            2: [],
            "crossed": [],
        }  # Reset vehicle dictionary
    
    # Reset waiting times
    for direction in waiting_times:
        waiting_times[direction] = {}  # Reset waiting times dictionary
    
    # Reset global crash counter
    crashes = 0
    
    # Re-initialize traffic signals to ensure clean state
    initialize()
    
    # Note: Vehicle generation thread will continue and repopulate
    logger.info("Simulation environment fully reset.")


def start_simulation_threads():
    """Startet die Hintergrunddienste für die Simulation."""
    # Initialize Traffic Signals
    initialize()
    
    # Start Vehicle Generation Thread
    vehicle_thread = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    vehicle_thread.daemon = True
    vehicle_thread.start()
    logger.info("Vehicle generation thread started")

    # Start initialization thread
    init_thread = threading.Thread(name="initialization", target=initialize, args=())
    init_thread.daemon = True
    init_thread.start()
    logger.info("Initialization thread started")
    
    return [False] * noOfSignals  # Start with all red lights


def calculate_metrics():
    """Berechnet alle Metriken für den aktuellen Simulationsschritt."""
    waiting_vehicles_this_step = 0
    sum_speed = 0
    vehicle_count = 0
    emissions_this_step = 0
    sum_sq_waiting_time = 0
    
    for vehicle in simulation:
        # Check waiting status
        if not vehicle.crashed and vehicle.waiting_time > 0:
            waiting_vehicles_this_step += 1

        # Sum speed for average calculation
        sum_speed += vehicle.speed
        vehicle_count += 1

        # Calculate instantaneous emission for this step
        if not vehicle.crashed:
            emissions_this_step += vehicle.update_emission()
            # Calculate sum of squared waiting times
            if vehicle.waiting_time > 0:
                sum_sq_waiting_time += vehicle.waiting_time**2
                
    # Calculate average speed
    avg_speed = sum_speed / vehicle_count if vehicle_count > 0 else 0
    
    return {
        "waiting_vehicles": waiting_vehicles_this_step,
        "avg_speed": avg_speed,
        "vehicle_count": vehicle_count,
        "emissions": emissions_this_step,
        "sum_sq_waiting_time": sum_sq_waiting_time
    }


def update_simulation(active_lights):
    """Führt einen Simulationsschritt aus und gibt Ergebnisse zurück."""
    crashes_this_step = 0
    vehicles_to_remove = []
    newly_crossed_count = 0

    # Move vehicles
    for vehicle in list(simulation):  # Iterate over a copy for safe removal
        crashes_result = 0
        if not vehicle.crashed:
            # Move the vehicle with the current light configuration
            crashes_result = vehicle.move(
                vehicles,
                active_lights,
                stopLines,
                movingGap,
                simulation,
            )
            crashes_this_step += crashes_result
            
            # Check if vehicle crossed the intersection during this move
            if vehicle.crossed == 1 and not hasattr(vehicle, "_crossed_last_step"):
                newly_crossed_count += 1
                vehicle._crossed_last_step = True  # Mark it to avoid counting again

        # Off-screen check with consistent screen dimensions
        off_screen = (
            (vehicle.direction == "right" and vehicle.x > SCREEN_WIDTH)
            or (vehicle.direction == "left" and vehicle.x + vehicle.width < 0)
            or (vehicle.direction == "down" and vehicle.y > SCREEN_HEIGHT)
            or (vehicle.direction == "up" and vehicle.y + vehicle.height < 0)
        )
        
        # Mark for removal if crashed OR off_screen
        if vehicle.crashed or off_screen:
            vehicles_to_remove.append(vehicle)

    # Remove marked vehicles
    for vehicle in vehicles_to_remove:
        simulation.remove(vehicle)
        try:
            vehicles[vehicle.direction][vehicle.lane].remove(vehicle)
            if vehicle.id in waiting_times[vehicle.direction]:
                del waiting_times[vehicle.direction][vehicle.id]
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Failed to remove vehicle {getattr(vehicle, 'id', '?')} from structure: {e}"
            )
    
    return {
        "crashes_this_step": crashes_this_step,
        "newly_crossed_count": newly_crossed_count,
        "removed_vehicles": len(vehicles_to_remove)
    }


def reset_agent(neural_controller):
    """Setzt den internen Zustand des Agenten zurück."""
    neural_controller.last_state = np.zeros(neural_controller.input_size)
    neural_controller.last_action = None
    neural_controller.last_action_log_prob = None
    neural_controller.last_value = None
    neural_controller.replay_buffer.clear()
    
    return neural_controller


def wait_for_space(neural_controller):
    """Wartet auf Drücken der Leertaste, um mit dem Training fortzufahren."""
    if not PERFORMANCE_MODE:
        logger.info("Waiting for SPACE key press to continue...")
        while neural_controller.waiting_for_space:
            if neural_controller.check_for_space():
                break  # Exit wait loop once space is pressed
            time.sleep(0.1)  # Prevent busy-waiting