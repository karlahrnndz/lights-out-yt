import cupy as cp
from typing import List
import random

# ============================================================
# Helper conversions between Python lists and CuPy arrays
# ============================================================

def grid_to_array(grid: List[List[int]]) -> cp.ndarray:
    """
    Convert a 2D Python list of 0/1 values into a CuPy boolean array.
    Working with booleans lets XOR act as addition in GF(2).
    """
    return cp.array(grid, dtype=bool)

def array_to_grid(arr: cp.ndarray) -> List[List[int]]:
    """
    Convert CuPy boolean array back into a Python list of 0/1 values.
    """
    return cp.asnumpy(arr).astype(int).tolist()

# ============================================================
# Build the Lights Out toggle (adjacency) matrix on GPU
# ============================================================

def build_toggle_matrix(rows: int, cols: int) -> cp.ndarray:
    """
    Construct the Lights Out toggle matrix A over GF(2) as a boolean CuPy array.
    Each row corresponds to pressing one button; each column to one light.
    """
    n = rows * cols
    matrix = cp.zeros((n, n), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c  # Flatten (r, c) to a single index

            # Pressing a button toggles itself
            matrix[idx, idx] = True

            # Toggle neighbors (up, down, left, right)
            if r > 0:
                matrix[idx, (r - 1) * cols + c] = True
            if r < rows - 1:
                matrix[idx, (r + 1) * cols + c] = True
            if c > 0:
                matrix[idx, r * cols + (c - 1)] = True
            if c < cols - 1:
                matrix[idx, r * cols + (c + 1)] = True

    return matrix

# ============================================================
# Gaussian elimination over GF(2) on GPU
# ============================================================

def gaussian_elimination(matrix: cp.ndarray, rhs: cp.ndarray) -> List[cp.ndarray]:
    """
    Solve A x = b over GF(2) using CuPy boolean arrays.
    Returns a list of all possible solutions (if any exist).
    """
    mat = matrix.copy()
    b = rhs.copy()
    n_rows, n_cols = mat.shape
    pivot_cols = []
    row = 0

    # ----------------------------
    # Forward elimination
    # ----------------------------
    for col in range(n_cols):
        pivot_row = None
        for r in range(row, n_rows):
            if mat[r, col]:
                pivot_row = r
                break
        if pivot_row is None:
            continue

        # Swap pivot row into place
        mat[[row, pivot_row]] = mat[[pivot_row, row]]
        b[[row, pivot_row]] = b[[pivot_row, row]]
        pivot_cols.append(col)

        # Eliminate this column from all other rows
        for r in range(n_rows):
            if r != row and mat[r, col]:
                mat[r] ^= mat[row]
                b[r] ^= b[row]
        row += 1

    # ----------------------------
    # Check for inconsistency
    # ----------------------------
    for r in range(row, n_rows):
        if not mat[r].any() and b[r]:
            return []

    # ----------------------------
    # Back-substitution
    # ----------------------------
    solution = cp.zeros(n_cols, dtype=bool)
    for i in reversed(range(len(pivot_cols))):
        col = pivot_cols[i]
        rhs_val = b[i]
        for j in range(i + 1, len(pivot_cols)):
            next_col = pivot_cols[j]
            if mat[i, next_col]:
                rhs_val ^= solution[next_col]
        solution[col] = rhs_val

    # ----------------------------
    # Enumerate all solutions
    # ----------------------------
    free_cols = set(range(n_cols)) - set(pivot_cols)
    all_solutions = [solution.copy()]
    for free_col in free_cols:
        new_solutions = []
        for sol in all_solutions:
            new_solutions.append(sol.copy())          # free_col = 0
            flipped = sol.copy()
            flipped[free_col] = True                  # free_col = 1
            new_solutions.append(flipped)
        all_solutions = new_solutions

    return all_solutions

# ============================================================
# High-level GPU solver
# ============================================================

def solve_lights_out(initial_grid: List[List[int]], final_grid: List[List[int]]) -> List[List[List[int]]]:
    rows, cols = len(initial_grid), len(initial_grid[0])

    init_arr = grid_to_array(initial_grid)
    final_arr = grid_to_array(final_grid)

    # Right-hand side: which lights need to change
    rhs = (init_arr ^ final_arr).reshape(rows * cols)

    # Build the toggle matrix
    matrix = build_toggle_matrix(rows, cols)

    # Solve the linear system on GPU
    solutions = gaussian_elimination(matrix, rhs)

    # Convert solutions back into grid form
    return [array_to_grid(sol.reshape(rows, cols)) for sol in solutions]

# ============================================================
# Example usage
# ============================================================

def random_grid(rows: int, cols: int) -> List[List[int]]:
    return [[random.choice((0, 1)) for _ in range(cols)] for _ in range(rows)]

def print_grid(grid: List[List[int]]) -> None:
    for row in grid:
        print(" ".join(str(cell) for cell in row))

if __name__ == "__main__":
    rows, cols = 5, 5

    initial_grid = [[1 for _ in range(cols)] for _ in range(rows)]
    final_grid = [[0] * cols for _ in range(rows)]  # target: all lights off

    print("Initial grid:")
    print_grid(initial_grid)

    solutions = solve_lights_out(initial_grid, final_grid)

    if not solutions:
        print("\nNo solution exists.")
    else:
        print(f"\n{len(solutions)} solution(s) found:")
        for i, sol in enumerate(solutions, 1):
            print(f"\nSolution {i}:")
            print_grid(sol)
