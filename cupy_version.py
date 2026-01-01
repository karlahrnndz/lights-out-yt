# gpu_based.py
import cupy as cp
from typing import List

def grid_to_gpu_bool(grid: List[List[int]]) -> cp.ndarray:
    """Convert 2D grid of 0/1 to a GPU boolean array."""
    return cp.array(grid, dtype=bool)

def gpu_bool_to_grid(arr: cp.ndarray) -> List[List[int]]:
    """Convert GPU boolean array back to 2D Python list."""
    return cp.asnumpy(arr).astype(int).tolist()

def build_toggle_matrix_gpu(rows: int, cols: int) -> cp.ndarray:
    """Build the Lights Out adjacency matrix as boolean GPU array."""
    n = rows * cols
    matrix = cp.zeros((n, n), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            matrix[idx, idx] = True
            if r > 0:
                matrix[idx, (r - 1) * cols + c] = True
            if r < rows - 1:
                matrix[idx, (r + 1) * cols + c] = True
            if c > 0:
                matrix[idx, r * cols + (c - 1)] = True
            if c < cols - 1:
                matrix[idx, r * cols + (c + 1)] = True
    return matrix

def gaussian_elimination_gpu(matrix: cp.ndarray, rhs: cp.ndarray) -> List[cp.ndarray]:
    """Solve matrix * x = rhs in GF(2) using GPU boolean arrays."""
    mat = matrix.copy()
    b = rhs.copy()
    n_rows, n_cols = mat.shape
    pivot_cols = []
    row = 0

    # Forward elimination
    for col in range(n_cols):
        pivot_row = None
        for r in range(row, n_rows):
            if mat[r, col]:
                pivot_row = r
                break
        if pivot_row is None:
            continue  # free variable
        # Swap pivot row into place
        mat[[row, pivot_row]] = mat[[pivot_row, row]]
        b[[row, pivot_row]] = b[[pivot_row, row]]
        pivot_cols.append(col)
        # Eliminate all other rows
        for r in range(n_rows):
            if r != row and mat[r, col]:
                mat[r] ^= mat[row]
                b[r] ^= b[row]
        row += 1

    # Check for inconsistency
    for r in range(row, n_rows):
        if not mat[r].any() and b[r]:
            return []  # no solution

    # Back-substitution (all free vars = 0)
    solution = cp.zeros(n_cols, dtype=bool)
    for i in reversed(range(len(pivot_cols))):
        col = pivot_cols[i]
        rhs_val = b[i]
        for j in range(i + 1, len(pivot_cols)):
            next_col = pivot_cols[j]
            if mat[i, next_col]:
                rhs_val ^= solution[next_col]
        solution[col] = rhs_val

    # Enumerate all solutions by flipping free variables
    free_cols = set(range(n_cols)) - set(pivot_cols)
    all_solutions = [solution.copy()]
    for free_col in free_cols:
        new_solutions = []
        for sol in all_solutions:
            new_solutions.append(sol.copy())          # free_col = 0
            sol2 = sol.copy()
            sol2[free_col] = True                     # free_col = 1
            new_solutions.append(sol2)
        all_solutions = new_solutions

    return all_solutions

def solve_w_gpu(initial_grid: List[List[int]], final_grid: List[List[int]]) -> List[List[List[int]]]:
    """High-level solver using GPU boolean arrays and XOR."""
    rows, cols = len(initial_grid), len(initial_grid[0])
    if len(final_grid) != rows or len(final_grid[0]) != cols:
        raise ValueError("Initial and final grids must have the same dimensions.")

    # Convert grids to GPU boolean arrays
    initial_gpu = grid_to_gpu_bool(initial_grid)
    final_gpu = grid_to_gpu_bool(final_grid)

    # Compute difference (GF(2) subtraction = XOR)
    rhs = initial_gpu.flatten() ^ final_gpu.flatten()

    # Build toggle matrix on GPU
    matrix_gpu = build_toggle_matrix_gpu(rows, cols)

    # Solve on GPU
    solutions_gpu = gaussian_elimination_gpu(matrix_gpu, rhs)

    # Convert solutions back to Python lists
    solutions = [gpu_bool_to_grid(sol.reshape(rows, cols)) for sol in solutions_gpu]
    return solutions
