from typing import List
import copy

def apply_solution(initial_grid: List[List[int]],
                   solution: List[List[int]]) -> List[List[int]]:
    """
    Apply a Lights Out solution to the initial grid.
    Pressing a cell toggles itself + von Neumann neighbors.
    """
    rows, cols = len(initial_grid), len(initial_grid[0])
    grid = copy.deepcopy(initial_grid)

    def toggle(r, c):
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] ^= 1  # XOR toggle

    for r in range(rows):
        for c in range(cols):
            if solution[r][c] == 1:
                toggle(r, c)
                toggle(r - 1, c)
                toggle(r + 1, c)
                toggle(r, c - 1)
                toggle(r, c + 1)

    return grid


def is_solution(initial_grid, final_grid, solution) -> bool:
    result = apply_solution(initial_grid, solution)
    return result == final_grid
