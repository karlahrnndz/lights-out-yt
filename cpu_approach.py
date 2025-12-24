from typing import List

def grid_to_int(grid: List[List[int]]) -> int:
    """Convert a 2D grid of 0/1 to a single integer (bit-packed)."""
    rows, cols = len(grid), len(grid[0])
    result = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                pos = r * cols + c
                result |= 1 << pos
    return result

def int_to_grid(x: int, rows: int, cols: int) -> List[List[int]]:
    """Convert integer back to 2D grid of size rows x cols."""
    grid = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            pos = r * cols + c
            if x & (1 << pos):
                grid[r][c] = 1
    return grid

def build_matrix(rows: int, cols: int) -> List[int]:
    """Build the Lights Out adjacency matrix (bit-packed rows)."""
    matrix = []
    for r in range(rows):
        for c in range(cols):
            row_mask = 0
            # toggle self
            row_mask |= 1 << (r * cols + c)
            # toggle neighbors
            if r > 0:
                row_mask |= 1 << ((r - 1) * cols + c)
            if r < rows - 1:
                row_mask |= 1 << ((r + 1) * cols + c)
            if c > 0:
                row_mask |= 1 << (r * cols + (c - 1))
            if c < cols - 1:
                row_mask |= 1 << (r * cols + (c + 1))
            matrix.append(row_mask)
    return matrix

def gaussian_elimination(matrix: List[int], rhs: List[int], num_cells: int) -> List[int]:
    """Solve matrix * x = rhs in Z2 using bit-packed integers."""
    n_rows = len(matrix)
    n_cols = num_cells
    mat = matrix[:]
    b = rhs[:]
    pivot_cols = []
    row = 0

    # Forward elimination
    for col in range(n_cols):
        pivot_row = None
        for r in range(row, n_rows):
            if (mat[r] >> col) & 1:
                pivot_row = r
                break
        if pivot_row is None:
            continue  # free variable
        # Swap pivot row into place
        mat[row], mat[pivot_row] = mat[pivot_row], mat[row]
        b[row], b[pivot_row] = b[pivot_row], b[row]
        pivot_cols.append(col)
        # Eliminate below and above
        for r in range(n_rows):
            if r != row and ((mat[r] >> col) & 1):
                mat[r] ^= mat[row]
                b[r] ^= b[row]
        row += 1

    # Check for inconsistency in zero rows
    for r in range(row, n_rows):
        if mat[r] == 0 and b[r]:
            return []  # no solution

    # Back-substitution to get one solution (all free vars = 0)
    solution = 0
    for i in reversed(range(len(pivot_cols))):
        col = pivot_cols[i]
        rhs_val = b[i]
        # XOR sum of all known variables to isolate pivot
        for j in range(i + 1, len(pivot_cols)):
            next_col = pivot_cols[j]
            if (mat[i] >> next_col) & 1:
                rhs_val ^= (solution >> next_col) & 1
        if rhs_val:
            solution |= 1 << col

    # Enumerate all solutions by flipping free variables
    free_cols = set(range(n_cols)) - set(pivot_cols)
    all_solutions = [solution]
    for free_col in free_cols:
        new_solutions = []
        for sol in all_solutions:
            new_solutions.append(sol)  # free_col = 0
            new_solutions.append(sol | (1 << free_col))  # free_col = 1
        all_solutions = new_solutions

    return all_solutions

def solve_lights_out(initial_grid: List[List[int]], final_grid: List[List[int]]) -> List[List[List[int]]]:
    rows, cols = len(initial_grid), len(initial_grid[0])
    if len(final_grid) != rows or len(final_grid[0]) != cols:
        raise ValueError("Initial and final grids must have the same dimensions.")

    initial_state = grid_to_int(initial_grid)
    final_state = grid_to_int(final_grid)
    rhs = [(initial_state ^ final_state) >> i & 1 for i in range(rows * cols)]

    matrix = build_matrix(rows, cols)
    solutions_int = gaussian_elimination(matrix, rhs, rows * cols)

    if not solutions_int:
        print("No solution exists.")
        return []

    # Convert integer solutions back to grids
    return [int_to_grid(sol, rows, cols) for sol in solutions_int]

# --- Example usage ---
if __name__ == "__main__":
    initial_grid = [
        [1,0],
        [0,0]
    ]
    final_grid = [
        [0,0],
        [0,0]
    ]

    solutions = solve_lights_out(initial_grid, final_grid)
    if solutions:
        print(f"{len(solutions)} solution(s) found:")
        for sol_grid in solutions:
            for row in sol_grid:
                print(row)
            print("---")
