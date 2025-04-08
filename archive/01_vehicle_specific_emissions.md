# Vehicle-Specific Emission Models

## Description
Currently, all vehicle types (cars, bikes, trucks, etc.) use the same emission penalty when accelerating. This is unrealistic as different vehicle types produce different amounts of CO2 emissions. We need to implement a vehicle-specific emission model that accounts for these differences.

## Current Implementation
- All vehicles increase the emission count by 1 when accelerating (in vehicle.py)
- Emission counts are tracked per direction but not per vehicle type
- The reward calculation uses a single EMISSION_FACTOR for all emissions

## Requirements
1. Modify the Vehicle class to include vehicle-type specific emission factors
2. Update the emission tracking mechanism to account for vehicle type
3. Adjust the reward calculation in the RL environment to use the new emission model

## Steps
1. Add vehicle-specific emission factors to the utils.py file
2. Modify the Vehicle class in vehicle.py to use type-specific emission factors when accelerating
3. Update the emission_counts tracking to store emissions by vehicle type
4. Modify the reward calculation in TrafficEnvironment._calculate_reward() to consider vehicle-specific emissions

## Expected Outcome
- Different vehicle types will have different emission impacts when accelerating
- Larger vehicles (trucks, buses) will generate more emissions than smaller ones (cars, bikes)
- The AI model will learn to prioritize signals to minimize overall emissions more effectively

## Status
Pending

## Priority
High

## Metadata
Created: 2025-04-08 