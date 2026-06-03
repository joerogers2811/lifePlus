from typing import List, Tuple
import numpy as np
from grid import Grid
from agent import Agent, Genome

class Simulation:
    """
    Orchestrator for the cellular automaton simulation.
    Handles the execution cycle (tick).
    """

    def __init__(self, grid: Grid, agents: List[Agent]):
        self.grid = grid
        self.agents = agents

    def tick(self):
        """
        Executes one simulation cycle consisting of three phases:
        1. Harvest: Gain resources and generate waste.
        2. Update: Update energy levels and apply waste penalties.
        3. Move & Reproduce: Change positions and create new agents.
        """
        # Phase 1: Harvest
        harvest_data = []
        for agent in self.agents:
            if not agent.alive:
                continue
            
            x, y = agent.position
            # Calculate potential resource gain based on cell's resource and agent's collection trait
            available_resource = self.grid.resource_a[x, y]
            gain = available_resource * agent.genome.energy_collection
            
            # Calculate waste generation (simplified: proportional to collection)
            waste_gen = gain * 0.5 
            
            harvest_data.append({
                "agent": agent,
                "gain": gain,
                "waste_gen": waste_gen,
                "pos": (x, y)
            })

        # Phase 2: Update
        for data in harvest_data:
            agent = data["agent"]
            x, y = data["pos"]
            
            # Apply gain
            agent.energy += data["gain"]
            
            # Update grid waste
            self.grid.waste_w[x, y] += data["waste_gen"]
            
            # Check waste penalty
            current_waste = self.grid.waste_w[x, y]
            if current_waste > agent.genome.waste_tolerance:
                # Immediate energy penalty
                penalty = (current_waste - agent.genome.waste_tolerance) * 0.2
                agent.energy -= penalty
            
            # Deplete cell resource (harvested)
            self.grid.resource_a[x, y] = max(0.0, self.grid.resource_a[x, y] - (data["gain"] / max(agent.genome.energy_collection, 0.1)))

            # Check survival
            if agent.energy <= 0:
                agent.alive = False
                self.grid.life_occupancy[x, y] = False

        # Phase 3: Move & Reproduce
        new_agents = []
        for agent in self.agents:
            if not agent.alive:
                continue
            
            old_x, old_y = agent.position
            
            # Decide Move using probabilistic scoring
            candidates = []
            speed = agent.genome.movement_speed
            for dx in range(-speed, speed + 1):
                for dy in range(-speed, speed + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = (old_x + dx) % self.grid.width
                    new_y = (old_y + dy) % self.grid.height
                    
                    if not self.grid.life_occupancy[new_x, new_y]:
                        candidates.append({
                            'pos': (new_x, new_y),
                            'resource': self.grid.resource_a[new_x, new_y],
                            'waste': self.grid.waste_w[new_x, new_y],
                            'distance': np.sqrt(dx**2 + dy**2)
                        })
            
            if candidates:
                new_pos = agent.calculate_move_probability(candidates)
                self.grid.life_occupancy[old_x, old_y] = False
                agent.position = new_pos
                self.grid.life_occupancy[new_pos[0], new_pos[1]] = True
                agent.energy -= 0.1 # Movement cost
            
            # Decide Reproduce (If enough energy)
            if agent.energy > 2.0:
                # Reproduce to an adjacent cell if available
                rx = (agent.position[0] + 1) % self.grid.width
                ry = agent.position[1]
                if not self.grid.life_occupancy[rx, ry]:
                    child_energy = 0.5
                    agent.energy -= (child_energy + 0.2) # Reproduction cost
                    
                    # Mutation logic integrated
                    child_genome = Agent.mutate_genome(agent.genome)
                    child = Agent((rx, ry), child_genome, energy=child_energy)
                    new_agents.append(child)
                    self.grid.life_occupancy[rx, ry] = True

        self.agents.extend(new_agents)
        # Cleanup dead agents
        self.agents = [a for a in self.agents if a.alive]
