from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors
import random


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        problem = self.problem
        best_state = None
        best_cost = float("inf")

        for run in range(self.max_restarts):
            current = problem.generate_initial_state()
            current = State(current.path, random.choice(problem.get_valid_time_domain()))
            nbrs = get_neighbors(current, problem)
            if not nbrs:
                break
            current = random.choice(nbrs)

            cur_cost = problem.evaluate(current)
            if cur_cost < best_cost:
                best_state, best_cost = current, cur_cost

            sideways = 0
            plateau_visited = {current}

            for _ in range(self.max_steps):
                neighbors = get_neighbors(current, problem)
                if not neighbors:
                    break

                score = [(problem.evaluate(nb), nb) for nb in neighbors]
                best_n_cost = min(c for c, _ in score)

                if best_n_cost < cur_cost:

                    ties = [nb for c, nb in score if c==best_n_cost]
                    current = random.choice(ties)
                    cur_cost = best_n_cost
                    sideways = 0
                    visited = {current}

                elif best_n_cost==cur_cost and sideways < self.max_sideways:
                    
                    ties = [
                        nb for c, nb in score
                        if c==cur_cost and nb not in visited
                    ]
                    if not ties:
                        break
                    current = random.choice(ties)
                    visited.add(current)
                    sideways += 1
                else:
                    break 

                if cur_cost < best_cost:
                    best_state, best_cost = current, cur_cost

        return best_state, best_cost
