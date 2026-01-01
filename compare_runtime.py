import time
import random
from typing import List

# from cupy_version import solve_with_cupy_gpu
# from galois_version import solve_with_galois
from lights_out import solve_lights_out
from numpy_numba import solve_with_numpy_numba


# -----------------------------
# Assume these functions exist:
# -----------------------------
# solve_with_galois(initial_grid, final_grid)
# solve_with_numpy_cpu(initial_grid, final_grid)
# solve_with_cupy_gpu(initial_grid, final_grid)

# Example placeholders:
# from version1_galois import solve_lights_out as solve_with_galois
# from version2_numpy import solve_lights_out as solve_with_numpy_cpu
# from version3_cupy import solve_lights_out as solve_with_cupy_gpu

def random_grid(rows: int, cols: int) -> List[List[int]]:
    """All lights on."""
    return [[1 for _ in range(cols)] for _ in range(rows)]

def time_solver(solver_func, sizes=list(range(10, 101, 10)), trials=2):
    results = {}
    for n in sizes:
        rows = cols = n
        total_time = 0.0
        for _ in range(trials):
            initial_grid = random_grid(rows, cols)
            final_grid = [[0] * cols for _ in range(rows)]  # all off
            start = time.time()
            solver_func(initial_grid, final_grid)
            end = time.time()
            total_time += (end - start)
        avg_time = total_time / trials
        results[n] = avg_time
        print(f"Size {n}x{n}: avg time over {trials} trials = {avg_time:.4f} s")
    return results

# -----------------------------
# Run timing tests
# -----------------------------

numpy_times = time_solver(solve_lights_out)
