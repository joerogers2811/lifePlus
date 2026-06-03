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

if __name__ == '__main__':
    unittest.main()
