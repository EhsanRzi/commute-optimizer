from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors
import random


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        problem = self.problem
        cache = {}

        base = problem.generate_initial_state()
        beam = [base]
        attempts = 0
        while len(beam) < self.beam_width:
            s = State(base.path, random.choice(problem.get_valid_time_domain()))
            nbrs = get_neighbors(s, problem)
            if nbrs:
                s = random.choice(nbrs)
            if s not in beam:
                beam.append(s)
            attempts += 1

        for s in beam:
            cache[s] = problem.evaluate(s)
        best_state = min(beam, key=lambda s: cache[s])
        best_cost = cache[best_state]
        self.history.append(best_cost)
        stale = 0

        for _ in range(self.max_iterations):
            pool = set(beam)
            for s in beam:
                pool.update(get_neighbors(s, problem))
            for s in pool:
                if s not in cache:
                    cache[s] = problem.evaluate(s)

            ranked = sorted(pool, key=lambda s: cache[s])

            selected = []
            leftovers = []
            per_path = {}
            for s in ranked:
                if len(selected) >= self.beam_width:
                    break
                
                selected.append(s)
                per_path[s.path] = per_path.get(s.path, 0) + 1

            new_best = min(selected, key=lambda s: cache[s])
            if cache[new_best] < best_cost:
                best_state, best_cost = new_best, cache[new_best]
                stale = 0
            else:
                stale += 1

            self.history.append(best_cost)
            beam = selected
            if stale >= 50:
                break

        return best_state, best_cost
