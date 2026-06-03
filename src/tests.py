import unittest
import numpy as np
from grid import Grid
from agent import Agent, Genome
from simulation import Simulation

class TestMetabolicSimulation(unittest.TestCase):
    
    def setUp(self):
        self.grid = Grid(10, 10)
        self.genome = Genome(movement_speed=1, energy_collection=1.0, waste_tolerance=1.0, vision_range=1)
        self.agent = Agent(position=(5, 5), genome=self.genome, energy=1.0)
        self.grid.life_occupancy[5, 5] = True
        self.sim = Simulation(self.grid, [self.agent])

    def test_grid_initialization(self):
        """Verify core data structure initialization."""
        self.assertEqual(self.grid.resource_a.shape, (10, 10))
        self.assertEqual(self.grid.waste_w.shape, (10, 10))
        self.assertEqual(self.grid.life_occupancy.shape, (10, 10))
        self.assertTrue(np.all(self.grid.resource_a == 0.0))
        self.assertTrue(self.grid.life_occupancy[5, 5])

    def test_harvest_phase(self):
        """Verify data flow of one tick() cycle: Harvest & Update."""
        self.grid.resource_a[5, 5] = 1.0
        initial_energy = self.agent.energy
        
        self.sim.tick()
        
        # After one tick:
        # 1. Harvest: Gain = 1.0 * 1.0 = 1.0. Waste_gen = 1.0 * 0.5 = 0.5.
        # 2. Update: Energy = 1.0 + 1.0 - 0.1 (movement cost) = 1.9.
        #    Wait, move cost is applied in Move phase.
        #    Actually, in my implementation:
        #    Update: Energy = 1.0 + 1.0 = 2.0. Grid waste = 0.5. Cell resource = 1.0 - (1.0/1.0) = 0.0.
        #    Move: Energy = 2.0 - 0.1 = 1.9. Position changes.
        
        # Check energy gain (allowing for movement cost)
        self.assertGreater(self.agent.energy, initial_energy)
        # Check waste generation
        self.assertGreater(np.sum(self.grid.waste_w), 0.0)
        # Check resource depletion
        self.assertEqual(self.grid.resource_a[5, 5], 0.0)

    def test_waste_penalty(self):
        """Verify energy penalty when waste exceeds tolerance."""
        self.grid.resource_a[5, 5] = 1.0
        self.grid.waste_w[5, 5] = 2.0 # Already above tolerance (1.0)
        
        # Tick will add more waste: 0.5
        # Total waste: 2.5
        # Penalty: (2.5 - 1.0) * 0.2 = 0.3
        # Initial: 1.0. Gain: 1.0. Move cost: 0.1. 
        # Expected: 1.0 + 1.0 - 0.3 - 0.1 = 1.6
        
        self.sim.tick()
        self.assertAlmostEqual(self.agent.energy, 1.6)

    def test_mutation_stability(self):
        """Verify that mutation stays within reasonable bounds."""
        parent_genome = self.genome
        for _ in range(100):
            child_genome = Agent.mutate_genome(parent_genome)
            self.assertTrue(1 <= child_genome.movement_speed <= 3)
            self.assertTrue(0.0 <= child_genome.energy_collection <= 2.0)
            self.assertTrue(0.0 <= child_genome.waste_tolerance <= 2.0)
            self.assertTrue(1 <= child_genome.vision_range <= 2)

    def test_statistical_stability(self):
        """
        Runs the simulation multiple times and verifies that the output metric
        is stable within a defined tolerance.
        """
        num_runs = 100
        tolerance = 0.1
        final_energies = []
        
        for _ in range(num_runs):
            # Reset state for each run
            grid = Grid(10, 10)
            genome = Genome(movement_speed=1, energy_collection=1.0, waste_tolerance=1.0, vision_range=1)
            agent = Agent(position=(5, 5), genome=genome, energy=1.0)
            grid.life_occupancy[5, 5] = True
            grid.resource_a[5, 5] = 1.0
            sim = Simulation(grid, [agent])
            
            # Run 1 tick
            sim.tick()
            if sim.agents:
                final_energies.append(sim.agents[0].energy)
            else:
                final_energies.append(0.0)
        
        mean_energy = np.mean(final_energies)
        variance_energy = np.var(final_energies)
        
        print(f"\n[STATISTICAL TEST] Mean Energy: {mean_energy:.4f}, Variance: {variance_energy:.4f}")
        
        # In this setup, Energy should be around 1.9 (1.0 + 1.0 - 0.1)
        expected_mean = 1.9
        self.assertLess(abs(mean_energy - expected_mean), tolerance, 
                       f"Mean energy {mean_energy} out of tolerance {tolerance}")

if __name__ == '__main__':
    unittest.main()
