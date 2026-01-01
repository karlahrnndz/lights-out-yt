# version1_galois.py
from galois import GF
from typing import List
import random

# ============================================================
# Build the Lights Out toggle (adjacency) matrix using GF(2)
# ============================================================

def build_toggle_matrix(rows: int, cols: int, GF2) -> GF2:
    """
    Construct the Lights Out toggle matrix A over GF(2).
    Each row corresponds to pressing one button; each column to one light.
    """
    n = rows * cols
    matrix = GF2.Zeros((n, n))

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c  # Flatten (r, c) to a single index

            # Pressing a button toggles itself
            matrix[idx, idx] = 1

            # Toggle neighbors (up, down, left, right)
            if r > 0:
                matrix[idx, (r - 1) * cols + c] = 1
            if r < rows - 1:
                matrix[idx, (r + 1) * cols + c] = 1
            if c > 0:
                matrix[idx, r * cols + (c - 1)] = 1
            if c < cols - 1:
                matrix[idx, r * cols + (c + 1)] = 1

    return matrix

# ============================================================
# High-level solver using Galois
# ============================================================

def solve_lights_out(initial_grid: List[List[int]], final_grid: List[List[int]]) -> List[List[List[int]]]:
    """
    Solve the Lights Out puzzle using Galois GF(2) linear algebra.
    Returns a list of all solutions (if any exist).
    """
    GF2 = GF(2)
    rows, cols = len(initial_grid), len(initial_grid[0])

    # Flatten grids and convert to GF(2)
    init_arr = GF2([cell for row in initial_grid for cell in row])
    final_arr = GF2([cell for row in final_grid for cell in row])

    # Right-hand side
    rhs = init_arr + final_arr  # addition in GF(2) is XOR

    # Build toggle matrix
    matrix = build_toggle_matrix(rows, cols, GF2)

    # Solve linear system
    try:
        # Solve using Galois; returns one solution if solvable
        solution = matrix.solve(rhs)

        # Convert solution to list of lists
        solution_grid = [solution[i * cols:(i + 1) * cols].tolist() for i in range(rows)]
        return [solution_grid]
    except Exception:
        # No solution exists
        return []

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
