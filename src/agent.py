from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Genome:
    """
    Defines the genetic traits of an agent.
    """
    movement_speed: int  # 1-3
    energy_collection: float  # 0-2
    waste_tolerance: float  # 0-2
    vision_range: int  # 1-2

class Agent:
    """
    An entity within the cellular automaton simulation.
    """

    def __init__(self, position: Tuple[int, int], genome: Genome, energy: float = 1.0):
        """
        Initializes an agent with a position, genome, and initial energy.
        """
        self.position = position
        self.genome = genome
        self.energy = energy
        self.alive = True

    def __repr__(self):
        return f"Agent(pos={self.position}, energy={self.energy:.2f})"
