import numpy as np
from typing import List
import random
import time


# ============================================================
# Helper functions
# ============================================================

def grid_to_array(grid: List[List[int]]) -> np.ndarray:
    return np.array(grid, dtype=bool)

def array_to_grid(arr: np.ndarray) -> List[List[int]]:
    return arr.astype(int).tolist()

def build_toggle_matrix(rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    mat = np.zeros((n, n), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            mat[idx, idx] = True
            if r > 0:
                mat[idx, (r - 1) * cols + c] = True
            if r < rows - 1:
                mat[idx, (r + 1) * cols + c] = True
            if c > 0:
                mat[idx, r * cols + (c - 1)] = True
            if c < cols - 1:
                mat[idx, r * cols + (c + 1)] = True
    return mat

# ============================================================
# Row reduction over GF(2)
# ============================================================

def row_reduce_gf2(matrix: np.ndarray, rhs: np.ndarray):
    mat = matrix.copy()
    b = rhs.copy()
    n_rows, n_cols = mat.shape
    pivot_cols = []
    row = 0

    for col in range(n_cols):
        pivot_row = None
        for r in range(row, n_rows):
            if mat[r, col]:
                pivot_row = r
                break
        if pivot_row is None:
            continue

        if pivot_row != row:
            mat[[row, pivot_row]] = mat[[pivot_row, row]]
            b[[row, pivot_row]] = b[[pivot_row, row]]

        pivot_cols.append(col)

        # Vectorized elimination
        mask = mat[:, col].copy()
        mask[row] = False
        mat[mask] ^= mat[row]
        b[mask] ^= b[row]

        row += 1

    # Inconsistency check
    for r in range(row, n_rows):
        if not mat[r].any() and b[r]:
            return pivot_cols, [], 0, mat, b

    free_cols = list(set(range(n_cols)) - set(pivot_cols))
    num_free = len(free_cols)
    num_solutions = 2 ** num_free

    return pivot_cols, free_cols, num_solutions, mat, b

# ============================================================
# Generate one solution
# ============================================================

def generate_one_solution(pivot_cols, reduced_matrix, reduced_rhs):
    n_cols = reduced_matrix.shape[1]
    solution = np.zeros(n_cols, dtype=bool)
    for i in reversed(range(len(pivot_cols))):
        col = pivot_cols[i]
        rhs_val = reduced_rhs[i]
        for j in range(i + 1, len(pivot_cols)):
            next_col = pivot_cols[j]
            if reduced_matrix[i, next_col]:
                rhs_val ^= solution[next_col]
        solution[col] = rhs_val
    return solution

# ============================================================
# Generate all solutions (careful: exponential!)
# ============================================================

def generate_all_solutions(pivot_cols, free_cols, reduced_matrix, reduced_rhs):
    one_sol = generate_one_solution(pivot_cols, reduced_matrix, reduced_rhs)
    all_sols = [one_sol.copy()]

    for free_col in free_cols:
        new_sols = []
        for sol in all_sols:
            new_sols.append(sol.copy())
            flipped = sol.copy()
            flipped[free_col] = True
            new_sols.append(flipped)
        all_sols = new_sols

    return all_sols

# ============================================================
# High-level solver
# ============================================================

def solve_lights_out(
    initial_grid: List[List[int]],
    final_grid: List[List[int]],
    find_sols: str = "one"  # "one"=default, "all", "none"
) -> dict:
    """
    Solve Lights Out.

    Args:
        find_sols: "one" → store at most one solution (default)
                   "all" → store all solutions
                   "none" → store no solutions

    Returns:
        dict with keys:
        - 'solutions': list of solutions or empty list if none/materialized=none
        - 'num_solutions': total number of solutions
    """
    rows, cols = len(initial_grid), len(initial_grid[0])
    n = rows * cols

    init_arr = grid_to_array(initial_grid)
    final_arr = grid_to_array(final_grid)
    rhs = (init_arr ^ final_arr).reshape(n)

    matrix = build_toggle_matrix(rows, cols)
    pivot_cols, free_cols, num_solutions, reduced_matrix, reduced_rhs = row_reduce_gf2(matrix, rhs)

    solutions = []

    if num_solutions == 0:
        return {'solutions': solutions, 'num_solutions': 0}

    if find_sols == "one":
        sol = generate_one_solution(pivot_cols, reduced_matrix, reduced_rhs)
        solutions.append(array_to_grid(sol.reshape(rows, cols)))
    elif find_sols == "all":
        sols = generate_all_solutions(pivot_cols, free_cols, reduced_matrix, reduced_rhs)
        solutions = [array_to_grid(sol.reshape(rows, cols)) for sol in sols]
    # find_sols == "none" → leave solutions empty

    return {'solutions': solutions, 'num_solutions': num_solutions}

# ============================================================
# Utilities
# ============================================================

def random_grid(rows: int, cols: int) -> List[List[int]]:
    return [[random.choice((0, 1)) for _ in range(cols)] for _ in range(rows)]

def print_grid(grid: List[List[int]]) -> None:
    for row in grid:
        print(" ".join(str(cell) for cell in row))

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    rows, cols = 30, 30
    initial_grid = [[1]*cols for _ in range(rows)]
    final_grid = [[0]*cols for _ in range(rows)]

    start = time.time()
    result = solve_lights_out(initial_grid, final_grid, find_sols="one")  # "one", "all", "none"
    end = time.time()
    print("Elapsed:", end - start, "seconds")

    if result['solutions']:
        print("\nSolutions materialized:", len(result['solutions']))
    else:
        print("\nNo solutions materialized.")

    print("Total number of solutions:", result['num_solutions'])
