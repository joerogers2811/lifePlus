from grid import Grid
from agent import Agent, Genome
from simulation import Simulation
import numpy as np

def main():
    # Initialize Grid
    grid = Grid(100, 100)
    
    # Set some initial resources
    grid.resource_a[50, 50] = 1.0
    grid.resource_a[51, 51] = 0.8
    
    # Initialize Agent
    genome = Genome(movement_speed=1, energy_collection=1.5, waste_tolerance=0.5, vision_range=1)
    agent = Agent(position=(50, 50), genome=genome, energy=1.0)
    grid.life_occupancy[50, 50] = True
    
    # Initialize Simulation
    sim = Simulation(grid, [agent])
    
    print(f"Initial State: {agent}")
    print(f"Cell (50, 50) Resource: {grid.resource_a[50, 50]}")
    
    # Run a few ticks
    for i in range(5):
        sim.tick()
        print(f"Tick {i+1}: {sim.agents}")
        # Replenish resources slightly for demo
        grid.resource_a += 0.05
        grid.resource_a = np.clip(grid.resource_a, 0.0, 1.0)

if __name__ == "__main__":
    main()
