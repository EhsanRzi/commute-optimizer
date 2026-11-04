from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors
import random


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)

    def solve(self) -> Tuple[State, float]:

        problem = self.problem

        current = problem.generate_initial_state()
        cur_cost = problem.evaluate(current)
        best_state, best_cost = current, cur_cost

        temp = self.initial_temp
        for _ in range(self.max_iterations):
            if temp < self.min_temp:
                break

            neighbors = get_neighbors(current, problem)
            if not neighbors:
                break

            candidate = random.choice(neighbors)
            cand_cost = problem.evaluate(candidate)
            delta = cand_cost - cur_cost

            if delta <= 0 or random.random() < math.exp(-delta/temp):
                current, cur_cost = candidate, cand_cost
                if cur_cost < best_cost:
                    best_state, best_cost = current, cur_cost

            temp *= self.cooling_rate

        return best_state, best_cost
