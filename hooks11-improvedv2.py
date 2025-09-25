import pprint
from collections import deque

# --- PART 1: BOARD AND SHAPE SETUP ---
def create_hook_map(size):
    """Create proper nested L-shaped hooks 1-9"""
    hook_map = [[0] * size for _ in range(size)]
    
    # Create hooks from outermost (9) to innermost (1)
    for hook_id in range(size, 0, -1):
        border_thickness = (size - hook_id) // 2
        inner_size = size - 2 * border_thickness
        
        if inner_size <= 0:
            continue
            
        # Fill the L-shape for this hook
        # Top edge of the L
        for c in range(border_thickness, border_thickness + inner_size):
            if hook_map[border_thickness][c] == 0:  # Don't overwrite
                hook_map[border_thickness][c] = hook_id
        
        # Left edge of the L (excluding corner already filled)
        for r in range(border_thickness + 1, border_thickness + inner_size):
            if hook_map[r][border_thickness] == 0:  # Don't overwrite
                hook_map[r][border_thickness] = hook_id
    
    return hook_map

def create_hook_lookup(hook_map, size):
    lookup = {}
    for r in range(size):
        for c in range(size):
            hook_id = hook_map[r][c]
            if hook_id not in lookup: 
                lookup[hook_id] = []
            lookup[hook_id].append((r, c))
    return lookup

def get_polyomino_shapes():
    """Standard 12 pentominoes - using only 9 distinct ones"""
    return {
        "F": {(0,1),(1,0),(1,1),(1,2),(2,1)},
        "I": {(0,0),(0,1),(0,2),(0,3),(0,4)},
        "L": {(0,0),(1,0),(2,0),(3,0),(3,1)},
        "N": {(0,1),(0,2),(1,0),(1,1),(2,0)},
        "P": {(0,0),(0,1),(1,0),(1,1),(2,0)},
        "T": {(0,0),(0,1),(0,2),(1,1),(2,1)},
        "U": {(0,0),(0,2),(1,0),(1,1),(1,2)},
        "V": {(0,0),(1,0),(2,0),(2,1),(2,2)},
        "W": {(0,0),(1,0),(1,1),(2,1),(2,2)},
        "X": {(0,1),(1,0),(1,1),(1,2),(2,1)},
        "Y": {(0,1),(1,0),(1,1),(1,2),(1,3)},
        "Z": {(0,0),(0,1),(1,1),(2,1),(2,2)}
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

# --- PART 2: DLX SOLVER AND TILING ---
class DLXSolver:
    def __init__(self, columns, matrix):
        self.header = self._build_links(columns, matrix)

    def _build_links(self, columns, matrix):
        header = Node('header')
        col_nodes = {name: Node(name) for name in columns}
        for name in columns:
            header.link_right(col_nodes[name])
        for row_name, row_cols in matrix.items():
            first_node = None
            for col_name in row_cols:
                new_node = Node(row_name, col_nodes[col_name])
                col_nodes[col_name].u.link_down(new_node)
                if first_node is None:
                    first_node = new_node
                else:
                    first_node.link_right(new_node)
        return header

    def solve(self):
        solution = self._search()
        if solution:
            return [{node.name} | {n.col.name for n in node.iter_right()} for node in solution]
        return None

    def _search(self):
        if self.header.r == self.header:
            return []
        c = self._choose_column()
        self.cover(c)
        for r in c.iter_down():
            for j in r.iter_right():
                self.cover(j.col)
            sub_solution = self._search()
            if sub_solution is not None:
                return [r] + sub_solution
            for j in r.iter_left():
                self.uncover(j.col)
        self.uncover(c)
        return None

    def _choose_column(self):
        min_size, chosen_col = float('inf'), None
        for col in self.header.iter_right():
            if col.size < min_size:
                min_size, chosen_col = col.size, col
        return chosen_col

    def cover(self, c):
        c.r.l, c.l.r = c.l, c.r
        for i in c.iter_down():
            for j in i.iter_right():
                j.u.d, j.d.u = j.d, j.u
                j.col.size -= 1

    def uncover(self, c):
        for i in c.iter_up():
            for j in i.iter_left():
                j.col.size += 1
                j.u.d, j.d.u = j, j
        c.r.l, c.l.r = c, c

class Node:
    def __init__(self, name, col=None):
        self.name = name
        self.col = col if col else self
        self.l, self.r, self.u, self.d = self, self, self, self
        self.size = 0 if col else 1

    def link_right(self, other):
        self.r, other.l = other, self

    def link_down(self, other):
        self.d, other.u = other, self
        self.col.size += 1

    def iter_right(self):
        curr = self.r
        while curr != self:
            yield curr
            curr = curr.r

    def iter_left(self):
        curr = self.l
        while curr != self:
            yield curr
            curr = curr.l

    def iter_down(self):
        curr = self.d
        while curr != self:
            yield curr
            curr = curr.d

    def iter_up(self):
        curr = self.u
        while curr != self:
            yield curr
            curr = curr.u

def can_be_tiled_dlx(board, required_shapes, size):
    target_cells = {(r, c) for r in range(size) for c in range(size) if board[r][c] != 0}
    total_shape_cells = sum(len(POLYOMINO_LIB[name]) for name, _ in required_shapes)
    if len(target_cells) != total_shape_cells: 
        return None

    columns = [name for name, _ in required_shapes] + list(target_cells)
    matrix = {}
    row_num = 0
    for name, orientations in required_shapes:
        for shape in orientations:
            for r in range(size):
                for c in range(size):
                    placed_coords = {(r + ro, c + co) for ro, co in shape}
                    if placed_coords.issubset(target_cells):
                        matrix[row_num] = [name] + list(placed_coords)
                        row_num += 1
    if not matrix: 
        return None
        
    solver = DLXSolver(columns, matrix)
    solution = solver.solve()
    if not solution: 
        return None
        
    tiling_solution = {}
    for row in solution:
        name = next(item for item in row if isinstance(item, str))
        coords = {item for item in row if isinstance(item, tuple)}
        tiling_solution[name] = coords
    return tiling_solution

# --- PART 3: VALIDATION FUNCTIONS ---
def is_connected(board, size):
    filled_cells = {(r, c) for r in range(size) for c in range(size) if board[r][c] != 0}
    if not filled_cells: 
        return True
    q = deque([next(iter(filled_cells))])
    visited = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in filled_cells and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return len(visited) == len(filled_cells)

def no_filled_2x2(board, size):
    for r in range(size - 1):
        for c in range(size - 1):
            if (board[r][c] != 0 and board[r+1][c] != 0 and 
                board[r][c+1] != 0 and board[r+1][c+1] != 0):
                return False
    return True

def is_pentomino_sum_valid(board, tiling_solution):
    if not tiling_solution:
        return False
    
    # Each pentomino must sum to a multiple of 5
    for shape_name, coords in tiling_solution.items():
        current_sum = sum(board[r][c] for r, c in coords)
        if current_sum % 5 != 0:
            return False
            
    # Get all the orientations for uniqueness check
    used_shapes = set()
    for shape_name, coords in tiling_solution.items():
        # Convert coords to normalized form for comparison
        min_r = min(r for r, c in coords)
        min_c = min(c for r, c in coords)
        normalized = frozenset((r - min_r, c - min_c) for r, c in coords)
        
        # Check all possible orientations and reflections
        all_orientations = []
        current = normalized
        for _ in range(4):  # 4 rotations
            # Add rotation
            all_orientations.append(current)
            # Add reflection of rotation
            reflected = frozenset((r, -c) for r, c in current)
            all_orientations.append(reflected)
            # Rotate 90 degrees
            current = frozenset((-c, r) for r, c in current)
            
        # If any orientation matches a used shape, it's not unique
        if any(orient in used_shapes for orient in all_orientations):
            return False
            
        used_shapes.add(normalized)
        
    return True

def is_hook_state_valid(board, hook_lookup, is_final_check=False):
    for hook_id in hook_lookup:
        hook_coords = hook_lookup[hook_id]
        hook_numbers = [board[r][c] for r, c in hook_coords if board[r][c] != -1]
        unique_digits = {n for n in hook_numbers if n != 0}
        
        # No hook can contain more than one type of digit
        if len(unique_digits) > 1: 
            return False
            
        if len(unique_digits) == 1:
            digit = unique_digits.pop()
            count = hook_numbers.count(digit)
            # Can't have more instances than the digit value
            if count > digit: 
                return False
            # In final check, must have exactly the right count
            if is_final_check and count != digit: 
                return False
    return True

def is_outside_clues_valid(board, clues, size, is_final_check=False):
    """Validate outside clues with proper intermediate/final logic"""
    
    # Check row clues
    for r_idx, clue_list in enumerate(clues["rows"]):
        if not clue_list:
            continue
            
        # Find first non-zero number when scanning left to right
        first_num = None
        first_pos = None
        for c in range(size):
            if board[r_idx][c] > 0:
                first_num = board[r_idx][c]
                first_pos = c
                break
        
        numeric_clues = [clue for clue in clue_list if isinstance(clue, int)]
        
        if is_final_check:
            # Final check: first number must match one of the numeric clues
            if numeric_clues and (first_num is None or first_num not in numeric_clues):
                return False
        else:
            # Intermediate check: if we see a number, ensure it's either valid or could be displaced
            if first_num is not None and numeric_clues:
                if first_num not in numeric_clues:
                    # Check if there's room to place a valid clue to the left
                    can_place_left = any(board[r_idx][c] == -1 for c in range(first_pos))
                    if not can_place_left:
                        return False

    # Check column clues
    for c_idx, clue_list in enumerate(clues["cols"]):
        if not clue_list:
            continue
            
        # Find first non-zero number when scanning top to bottom
        first_num = None
        first_pos = None
        for r in range(size):
            if board[r][c_idx] > 0:
                first_num = board[r][c_idx]
                first_pos = r
                break
        
        numeric_clues = [clue for clue in clue_list if isinstance(clue, int)]
        
        if is_final_check:
            # Final check: first number must match one of the numeric clues
            if numeric_clues and (first_num is None or first_num not in numeric_clues):
                return False
        else:
            # Intermediate check: if we see a number, ensure it's either valid or could be displaced
            if first_num is not None and numeric_clues:
                if first_num not in numeric_clues:
                    # Check if there's room to place a valid clue above
                    can_place_above = any(board[r][c_idx] == -1 for r in range(first_pos))
                    if not can_place_above:
                        return False

    return True

def solve_numbers(board, hook_lookup, required_shapes, clues, size):
    # Find next empty cell
    next_cell = None
    for r in range(size):
        for c in range(size):
            if board[r][c] == -1: 
                next_cell = (r, c)
                break
        if next_cell: 
            break

    # Base case: no more empty cells
    if not next_cell:
        # Convert -1s to 0s for final validation
        temp_board = [row[:] for row in board]
        for r in range(size):
            for c in range(size):
                if temp_board[r][c] == -1: 
                    temp_board[r][c] = 0

        # Check if we have the right number of filled cells
        required_filled_cells = sum(len(POLYOMINO_LIB[name]) for name, _ in required_shapes)
        actual_filled_cells = sum(1 for r in range(size) for c in range(size) if temp_board[r][c] > 0)
        
        if actual_filled_cells != required_filled_cells:
            return False

        # Run all final validations
        if not (is_hook_state_valid(temp_board, hook_lookup, is_final_check=True) and
                is_outside_clues_valid(temp_board, clues, size, is_final_check=True) and
                is_connected(temp_board, size) and
                no_filled_2x2(temp_board, size)):
            return False

        # Try to tile with pentominoes
        tiling_solution = can_be_tiled_dlx(temp_board, required_shapes, size)
        if not tiling_solution:
            return False

        # Validate pentomino constraints
        if not is_pentomino_sum_valid(temp_board, tiling_solution):
            return False

        # Success! Copy solution back
        for i in range(size): 
            board[i] = temp_board[i][:]
        return True

    # Recursive case: try placing numbers 0 through size
    r, c = next_cell
    for num in range(size + 1):
        board[r][c] = num
        
        # Check intermediate constraints
        if (is_hook_state_valid(board, hook_lookup) and 
            is_outside_clues_valid(board, clues, size, is_final_check=False)):
            
            if solve_numbers(board, hook_lookup, required_shapes, clues, size):
                return True
        
        # Backtrack
        board[r][c] = -1
    return False

# --- MAIN EXECUTION ---
REQUIRED_SHAPE_NAMES = ["F", "I", "L", "N", "P", "T", "U", "V", "W"]  # 9 standard pentominoes
grid_size = 9

# Fixed precondition grid that matches outside clues
precondition_grid = [
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3 matches col clue
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6 matches row clue
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Keep center 1
    [2, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 matches row clue
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7 matches col clue
    [0, 0, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 0, 0],
]

outside_clues = {
    "rows": [
        [], [], [], [6], [], [2], [], [], []
    ],
    "cols": [
        [], [], [3], [], [], [], [7], [], []
    ],
}

POLYOMINO_LIB = get_polyomino_shapes()
required_shapes = []
for name in REQUIRED_SHAPE_NAMES:
    if name not in POLYOMINO_LIB: 
        raise ValueError(f"Shape '{name}' not found.")
    required_shapes.append((name, get_all_orientations(POLYOMINO_LIB[name])))

# Initialize board
board = [[-1] * grid_size for _ in range(grid_size)]
for r in range(grid_size):
    for c in range(grid_size):
        if precondition_grid[r][c] != 0: 
            board[r][c] = precondition_grid[r][c]

# Create hook structure
hook_map = create_hook_map(grid_size)
hook_lookup = create_hook_lookup(hook_map, grid_size)

# Debug: Print hook structure
print("Hook map structure:")
for row in hook_map:
    print(row)
print(f"\nHook sizes: {[(hook_id, len(coords)) for hook_id, coords in sorted(hook_lookup.items())]}")

print(f"\nAttempting to solve. Requires tiling {len(required_shapes)} shapes.")
print(f"Need {len(required_shapes) * 5} filled cells total.")

if solve_numbers(board, hook_lookup, required_shapes, outside_clues, grid_size):
    print("\nSolution found!")
    pprint.pprint(board)
else:
    print("\nNo solution was found")