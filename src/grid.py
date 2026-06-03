import numpy as np

class Grid:
    """
    Central data store for the simulation.
    Manages state layers using NumPy arrays.
    """

    def __init__(self, width: int = 100, height: int = 100):
        """
        Initializes the grid with three layers:
        - Resource_A: [0.0, 1.0]
        - Waste_W: > 0.0
        - Life_Occupancy: Boolean
        """
        self.width = width
        self.height = height
        
        # Layer 0: Resource_A (float [0.0, 1.0])
        self.resource_a = np.zeros((width, height), dtype=np.float64)
        
        # Layer 1: Waste_W (float > 0.0)
        self.waste_w = np.zeros((width, height), dtype=np.float64)
        
        # Layer 2: Life_Occupancy (bool)
        self.life_occupancy = np.zeros((width, height), dtype=bool)

    def get_cell_state(self, x: int, y: int) -> dict:
        """
        Returns the state of a specific cell as a dictionary.
        """
        return {
            "Resource_A": self.resource_a[x, y],
            "Waste_W": self.waste_w[x, y],
            "Life_Occupancy": self.life_occupancy[x, y]
        }

    def update_cell(self, x: int, y: int, resource: float = None, waste: float = None, occupancy: bool = None):
        """
        Updates the values of a specific cell.
        """
        if resource is not None:
            self.resource_a[x, y] = np.clip(resource, 0.0, 1.0)
        if waste is not None:
            self.waste_w[x, y] = max(0.0, waste)
        if occupancy is not None:
            self.life_occupancy[x, y] = occupancy
