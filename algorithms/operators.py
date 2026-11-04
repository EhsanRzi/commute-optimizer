from typing import List
from core.state import State
from core.problem import CommuteProblem


def get_neighbors(state: State, problem: CommuteProblem) -> List[State]:

    path = state.path
    t = state.departure_time
    step = problem.t_step
    net = problem.network
    n = len(path)

    path_variants = []
    seen_paths = {path}
    max =8
    for i in range(n-1):
        for j in range(i+1, n):
            src, dst = path[i], path[j]
            blocked = set(path[:i]) | set(path[j + 1:])
            original = path[i:j+1]
            max_hops = (j-i)+3
            found = 0

            cur = [src]
            visited = {src}
            stack = [iter(net.get_neighbors(src))]
            while stack and found < max:
                advanced = False
                for v in stack[-1]:
                    if v in visited or v in blocked:
                        continue
                    if v == dst:
                        found += 1
                        sub = tuple(cur) + (v,)
                        if sub != original:
                            new_path = tuple(path[:i]) + sub + tuple(path[j + 1:])
                            if new_path not in seen_paths:
                                seen_paths.add(new_path)
                                path_variants.append(new_path)
                        if found >= max:
                            break
                        continue
                    if len(cur) < max_hops:
                        cur.append(v)
                        visited.add(v)
                        stack.append(iter(net.get_neighbors(v)))
                        advanced = True
                        break
                if found >= max:
                    break
                if not advanced:
                    stack.pop()
                    visited.discard(cur.pop())

    candidates = []
    
    candidates.append((path, t + step))
    candidates.append((path, t - step))

    for p in path_variants:
        candidates.append((p, t))

    for p in path_variants:
        candidates.append((p, t + step))
        candidates.append((p, t - step))

    neighbors = []
    seen = {state}
    for p, tt in candidates:
        if tt < problem.t_min or tt > problem.t_max:
            continue
        cand = State(p, round(tt, 2))
        if cand in seen:
            continue
        if not problem.is_valid_state(cand):
            continue
        seen.add(cand)
        neighbors.append(cand)

    return neighbors


def crossover(parent1: State, parent2: State, problem: CommuteProblem) -> State:

    p1, p2 = parent1.path, parent2.path

    child_time = (parent1.departure_time + parent2.departure_time)/2
    child_time = min(max(child_time, problem.t_min), problem.t_max)
    child_time = round(child_time, 2)

    v2 = {node: i for i, node in enumerate(p2)}
    shared = [
        (i, v2[node])
        for i, node in enumerate(p1)
        if 0 < i < len(p1)-1 and node in v2 and 0 < v2[node] < len(p2)-1
    ]

    for i1, i2 in shared:
        options = [
            tuple(p1[:i1+1]) + tuple(p2[i2+1:]),
            tuple(p2[:i2+1]) + tuple(p1[i1+1:]),
        ]
        for cp in options:
            child = State(cp, child_time)
            if problem.is_valid_state(child):
                return child

    base = parent1
    return base