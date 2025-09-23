import pprint

def create_hook_map():
    size = 9
    hook_map = [[0] * size for _ in range(size)]
    for i in range(size // 2 + 1):
        hook_id = size - 2 * i
        if hook_id < 1: continue
        for j in range(i, size - i):
            hook_map[i][j] = hook_id
            hook_map[j][i] = hook_id
    return hook_map

def create_hook_lookup(hook_map):
    lookup = {}
    for r in range(9):
        for c in range(9):
            hook_id = hook_map[r][c]
            if hook_id not in lookup:
                lookup[hook_id] = []
            lookup[hook_id].append((r, c))
    return lookup

def get_polyomino_shapes():
    return {
        "F": {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}, "I": {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)},
        "L": {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)}, "P": {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)},
        "N": {(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)}, "T": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
        "U": {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}, "V": {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
        "W": {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)}, "X": {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        "Y": {(0, 1), (1, 0), (1, 1), (1, 2), (1, 3)}, "Z": {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
    }

def get_all_orientations(shape):
    orientations = set()
    current_shape = shape
    for _ in range(4):
        orientations.add(tuple(sorted(list(current_shape))))
        flipped_shape = frozenset({(r, -c) for r, c in current_shape})
        orientations.add(tuple(sorted(list(flipped_shape))))
        current_shape = frozenset({(-c, r) for r, c in current_shape})
    return [set(o) for o in orientations]

def can_be_tiled(board, required_pentominoes):
    target_cells = sorted([ (r, c) for r in range(9) for c in range(9) if board[r][c] != 0 ])

    if len(target_cells) != len(required_pentominoes) * 5:
        return None

    def solve_tiling(cells_to_cover, shapes_to_place, solution):
        if not shapes_to_place:
            return solution

        shape_name, orientations = shapes_to_place[0]
        first_r, first_c = cells_to_cover[0]

        for shape in orientations:
            for r_offset, c_offset in shape:
                top_r, top_c = first_r - r_offset, first_c - c_offset
                placed_coords = {(top_r + ro, top_c + co) for ro, co in shape}
                
                if placed_coords.issubset(cells_to_cover):
                    remaining_cells = [cell for cell in cells_to_cover if cell not in placed_coords]
                    new_solution = solution.copy()
                    new_solution[shape_name] = placed_coords
                    
                    result = solve_tiling(remaining_cells, shapes_to_place[1:], new_solution)
                    if result is not None:
                        return result
        return None

    return solve_tiling(target_cells, required_pentominoes, {})

def check_pentomino_clues(tiling_solution, clues):
    if not tiling_solution: return False
    
    for r_idx, clue_list in enumerate(clues["rows"]):
        for clue in clue_list:
            if isinstance(clue, str):
                if clue not in tiling_solution: return False
                if not any(r == r_idx for r, c in tiling_solution[clue]):
                    return False
    for c_idx, clue_list in enumerate(clues["cols"]):
        for clue in clue_list:
            if isinstance(clue, str):
                if clue not in tiling_solution: return False
                if not any(c == c_idx for r, c in tiling_solution[clue]):
                    return False
    return True

def is_valid_number_placement(board, hook_lookup, clues):
    for hook_id in hook_lookup:
        count = sum(1 for r, c in hook_lookup[hook_id] if board[r][c] == hook_id)
        if count != hook_id:
            return False
            
    for r_idx, clue_list in enumerate(clues["rows"]):
        for clue in clue_list:
            if isinstance(clue, int):
                if clue not in board[r_idx]: return False
            
    for c_idx, clue_list in enumerate(clues["cols"]):
        col_vals = [board[r][c_idx] for r in range(9)]
        for clue in clue_list:
            if isinstance(clue, int):
                if clue not in col_vals: return False
            
    return True

def find_best_empty_cell(board, hook_map, hook_lookup):
    best_cell = None
    min_options = 11

    for r in range(9):
        for c in range(9):
            if board[r][c] == -1:
                options_count = 0
                # Count how many numbers (0-9) are a valid placement
                for num in range(10):
                    # Check if placing 'num' would immediately violate the hook rule
                    cell_hook_id = hook_map[r][c]
                    
                    # Temporarily place to check
                    board[r][c] = num
                    count = sum(1 for row, col in hook_lookup[cell_hook_id] if board[row][col] == cell_hook_id)
                    board[r][c] = -1 # Undo placement
                    
                    if count <= cell_hook_id:
                        options_count += 1
                
                # If this cell is the most constrained so far, it's our new best choice
                if options_count < min_options:
                    min_options = options_count
                    best_cell = (r, c)
                    # If a cell has 0 options, the board is unsolvable from this state
                    if min_options == 0:
                        return None
                        
    return best_cell

def solve_numbers(board, hook_map, hook_lookup, required_pentominoes, clues):
    next_cell = find_best_empty_cell(board, hook_map, hook_lookup)

    # If no valid empty cell is found, either we are done or we've hit a dead end
    if not next_cell:
        # Check if the board is actually full (no -1s left)
        is_full = not any(-1 in row for row in board)
        if is_full and is_valid_number_placement(board, hook_lookup, clues):
            print("Found a valid number placement. Checking if it can be tiled...")
            tiling_solution = can_be_tiled(board, required_pentominoes)
            if tiling_solution:
                print("Found a valid tiling. Checking pentomino clues...")
                if check_pentomino_clues(tiling_solution, clues):
                    return True
        return False

    r, c = next_cell
    for num in range(10):
        board[r][c] = num
        
        # Check if this move is consistent before recursing
        cell_hook_id = hook_map[r][c]
        count = sum(1 for row, col in hook_lookup[cell_hook_id] if board[row][col] == cell_hook_id)
        
        if count <= cell_hook_id:
            if solve_numbers(board, hook_map, hook_lookup, required_pentominoes, clues):
                return True

    board[r][c] = -1
    return False

REQUIRED_PENTOMINO_NAMES = ["I", "U", "N", "X", "V", "Z"]
precondition_grid = [
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 0, 0],
]

outside_clues = {
    "rows": [
        [1, "U"],
        [], 
        [], 
        [6, "X"], 
        [], 
        [2, "N"], 
        [], 
        [], 
        [9, "V"]
    ], 
    "cols": [
        [], 
        [], 
        [3], 
        [], 
        [], 
        [], 
        [7], 
        ["I"], 
        []
    ],
}
POLYOMINO_LIB = get_polyomino_shapes()
required_pentominoes = []
for name in REQUIRED_PENTOMINO_NAMES:
    if name not in POLYOMINO_LIB: raise ValueError(f"Shape '{name}' not found.")
    required_pentominoes.append((name, get_all_orientations(POLYOMINO_LIB[name])))

board = [[-1] * 9 for _ in range(9)]
for r in range(9):
    for c in range(9):
        if precondition_grid[r][c] != 0:
            board[r][c] = precondition_grid[r][c]

hook_map = create_hook_map()
hook_lookup = create_hook_lookup(hook_map)

print(f"Attempting to solve. Requires tiling {len(REQUIRED_PENTOMINO_NAMES)} pentominoes.")
if solve_numbers(board, hook_map, hook_lookup, required_pentominoes, outside_clues):
    print("\nSolution found! ✅")
    pprint.pprint(board)
else:
    print("\nNo solution was found. ❌")