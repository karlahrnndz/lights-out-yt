import numpy as np
from typing import List
import random

# ============================================================
# Helper conversions between Python lists and NumPy arrays
# ============================================================

def grid_to_array(grid: List[List[int]]) -> np.ndarray:
    """
    Convert a 2D Python list of 0/1 values into a NumPy boolean array.
    Working with booleans lets XOR act as addition in GF(2).
    """
    return np.array(grid, dtype=bool)

def array_to_grid(arr: np.ndarray) -> List[List[int]]:
    """
    Convert a NumPy boolean array back into a Python list of 0/1 values.
    """
    return arr.astype(int).tolist()

# ============================================================
# Build the Lights Out toggle (adjacency) matrix
# ============================================================

def build_toggle_matrix(rows: int, cols: int) -> np.ndarray:
    """
    Construct the Lights Out toggle matrix A over GF(2).

    Each row corresponds to pressing one button.
    Each column corresponds to one light on the board.

    A[i, j] = 1 if pressing button i toggles light j.
    """
    n = rows * cols
    matrix = np.zeros((n, n), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c  # Flatten (r, c) to a single index

            # Pressing a button always toggles itself
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
# Gaussian elimination over GF(2)
# ============================================================

def gaussian_elimination(matrix: np.ndarray, rhs: np.ndarray) -> List[np.ndarray]:
    """
    Solve A x = b over GF(2) using Gaussian elimination.

    - Addition is XOR
    - Multiplication is AND
    - Division is implicit because the only nonzero pivot is 1

    Returns:
        A list of all possible solutions (if any exist).
    """
    # Make copies so we don't modify inputs
    mat = matrix.copy()
    b = rhs.copy()

    n_rows, n_cols = mat.shape
    pivot_cols = []  # Track which columns have pivots
    row = 0          # Current pivot row

    # ----------------------------
    # Forward elimination
    # ----------------------------
    for col in range(n_cols):
        # Find a pivot row with a 1 in the current column
        pivot_row = None
        for r in range(row, n_rows):
            if mat[r, col]:
                pivot_row = r
                break

        # If no pivot exists, this column is a free variable
        if pivot_row is None:
            continue

        # Swap the pivot row into position
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
    # A row of all zeros with RHS = 1 means no solution
    for r in range(row, n_rows):
        if not mat[r].any() and b[r]:
            return []

    # ----------------------------
    # Back-substitution
    # ----------------------------
    # Construct one solution by setting all free variables to 0
    solution = np.zeros(n_cols, dtype=bool)

    for i in reversed(range(len(pivot_cols))):
        col = pivot_cols[i]
        rhs_val = b[i]

        # Subtract (XOR) already-known variables
        for j in range(i + 1, len(pivot_cols)):
            next_col = pivot_cols[j]
            if mat[i, next_col]:
                rhs_val ^= solution[next_col]

        solution[col] = rhs_val

    # ----------------------------
    # Enumerate all solutions
    # ----------------------------
    # Free variables can be flipped arbitrarily
    free_cols = set(range(n_cols)) - set(pivot_cols)
    all_solutions = [solution.copy()]

    for free_col in free_cols:
        new_solutions = []
        for sol in all_solutions:
            new_solutions.append(sol.copy())  # free_col = 0
            flipped = sol.copy()
            flipped[free_col] = True          # free_col = 1
            new_solutions.append(flipped)
        all_solutions = new_solutions

    return all_solutions

# ============================================================
# High-level solver
# ============================================================

def solve_with_numpy_cpu(
    initial_grid: List[List[int]],
    final_grid: List[List[int]]
) -> List[List[List[int]]]:
    """
    Solve the Lights Out puzzle by reducing it to A x = b over GF(2).
    """
    rows, cols = len(initial_grid), len(initial_grid[0])

    init_arr = grid_to_array(initial_grid)
    final_arr = grid_to_array(final_grid)

    # Right-hand side: which lights need to change
    rhs = (init_arr ^ final_arr).reshape(rows * cols)

    # Build the toggle matrix
    matrix = build_toggle_matrix(rows, cols)

    # Solve the linear system
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

    # initial_grid = random_grid(rows, cols)
    initial_grid = [[1 for _ in range(cols)] for _ in range(rows)]
    final_grid = [[0] * cols for _ in range(rows)]  # target: all lights off

    print("Initial grid:")
    print_grid(initial_grid)

    solutions = solve_with_numpy_cpu(initial_grid, final_grid)

    if not solutions:
        print("\nNo solution exists.")
    else:
        print(f"\n{len(solutions)} solution(s) found:")
        for i, sol in enumerate(solutions, 1):
            print(f"\nSolution {i}:")
            print_grid(sol)
