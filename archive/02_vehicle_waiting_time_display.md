# Vehicle Waiting Time Display

## Description
Currently, the simulation does not visually indicate how long each vehicle has been waiting. Adding a numerical display on top of each vehicle to show its waiting time would provide valuable visual feedback for both users and developers monitoring the simulation.

## Current Implementation
- The Vehicle class tracks waiting_time as an attribute
- This data is used for reward calculation but is not displayed in the UI
- Vehicles are rendered using their image without additional information

## Requirements
1. Add a visual indicator of waiting time on top of each vehicle in the simulation
2. The indicator should update in real-time as the vehicle waits
3. The indicator should be visible and easily readable

## Steps
1. Modify the Vehicle.render() method to display the waiting time
2. Create a font rendering method that draws text on top of each vehicle
3. Add logic to only show the waiting time when it's greater than 0
4. Make sure the text is visible against different vehicle colors

## Expected Outcome
- Each vehicle will display a number on top indicating how long it has been waiting
- The number will update in real-time as the vehicle waits
- When the vehicle is moving, the waiting time number will disappear
- This will provide a visual representation of traffic congestion and waiting patterns

## Status
Pending

## Priority
Medium

## Metadata
Created: 2025-04-08 