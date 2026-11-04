from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover
import random

class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    def solve(self) -> Tuple[State, float]:
        problem = self.problem
        cache = {}  

        base = problem.generate_initial_state()
        population = [base]
        while len(population) < self.population_size:
            s = State(base.path, random.choice(problem.get_valid_time_domain()))
            nbrs = get_neighbors(s, problem)
            if not nbrs:
                break
            s = random.choice(nbrs)
            population.append(s)

        for s in population:
            if s not in cache:
                cache[s] = problem.evaluate(s)

        best_state = min(population, key=lambda s: cache[s])
        best_cost = cache[best_state]

        elites = self.elitism

        for _ in range(self.generations):
            ranked = sorted(population, key=lambda s: cache[s])
            next_pop = list(ranked[:elites])

            while len(next_pop) < self.population_size:
                k = min(self.tournament_size, len(population))
                p1 = max(random.sample(population, k), key=lambda s: 1.0 / (1.0 + cache[s]))
                p2 = max(random.sample(population, k), key=lambda s: 1.0 / (1.0 + cache[s]))

                if random.random() < self.crossover_rate:
                    child = crossover(p1, p2, problem)
                else:
                    child = p1

                if random.random() < self.mutation_rate:
                    nbrs = get_neighbors(child, problem)
                    if nbrs:
                        child = random.choice(nbrs)

                if child not in cache:
                    cache[child] = problem.evaluate(child)
                next_pop.append(child)

            population = next_pop

            gen_best = min(population, key=lambda s: cache[s])
            if cache[gen_best] < best_cost:
                best_state, best_cost = gen_best, cache[gen_best]

        return best_state, best_cost
