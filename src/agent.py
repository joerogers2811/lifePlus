import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass(frozen=True)
class Genome:
    """
    Defines the genetic traits of an agent.
    """
    movement_speed: int  # 1-3
    energy_collection: float  # 0-2
    waste_tolerance: float  # 0-2
    vision_range: int  # 1-2
    # Decision weights for stochastic action scoring
    w_resource: float = 1.0
    w_waste: float = 0.5
    w_distance: float = 0.2

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

    @staticmethod
    def mutate_genome(parent_genome: Genome) -> Genome:
        """
        Creates a mutated version of the parent genome.
        Continuous traits are adjusted by a small symmetric distribution.
        Discrete traits are adjusted by small jumps within bounds.
        """
        def mutate_float(val: float, low: float, high: float) -> float:
            adjustment = np.random.uniform(-0.1, 0.1)
            return float(np.clip(val * (1 + adjustment), low, high))

        def mutate_int(val: int, low: int, high: int) -> int:
            adjustment = np.random.choice([-1, 0, 1])
            return int(np.clip(val + adjustment, low, high))

        return Genome(
            movement_speed=mutate_int(parent_genome.movement_speed, 1, 3),
            energy_collection=mutate_float(parent_genome.energy_collection, 0.0, 2.0),
            waste_tolerance=mutate_float(parent_genome.waste_tolerance, 0.0, 2.0),
            vision_range=mutate_int(parent_genome.vision_range, 1, 2),
            w_resource=mutate_float(parent_genome.w_resource, 0.0, 2.0),
            w_waste=mutate_float(parent_genome.w_waste, 0.0, 2.0),
            w_distance=mutate_float(parent_genome.w_distance, 0.0, 2.0)
        )

    def calculate_move_probability(self, candidates: List[Dict]) -> Tuple[int, int]:
        """
        Calculates the probability for each potential move using a softmax scoring function.
        Score = (W1 * Resource_A) + (W2 * WTolerance) - (W3 * Distance)
        Returns the chosen position based on a weighted random roll.
        """
        if not candidates:
            return self.position

        scores = []
        for c in candidates:
            # Score Neighbor N = (W1 * Resource_AN) + (W2 * WToleranceN) - (W3 * DistanceN)
            # Note: WToleranceN refers to the "Waste_W" layer compared to agent's tolerance, 
            # but prompt says (W2 * WToleranceN). 
            # Interpreting WToleranceN as (agent.waste_tolerance - cell_waste) or similar.
            # Actually, Prompt says: Score = (W1 * Resource_AN) + (W2 * WToleranceN) - (W3 * DistanceN)
            # Let's use the traits/values as specified.
            
            score = (self.genome.w_resource * c['resource']) + \
                    (self.genome.w_waste * (self.genome.waste_tolerance - c['waste'])) - \
                    (self.genome.w_distance * c['distance'])
            scores.append(score)

        # Softmax: P(N) = exp(ScoreN) / sum(exp(Scorek))
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores)) # Stability trick
        probabilities = exp_scores / np.sum(exp_scores)

        chosen_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[chosen_idx]['pos']

    def __repr__(self):
        return f"Agent(pos={self.position}, energy={self.energy:.2f})"
