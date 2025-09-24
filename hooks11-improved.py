import pprint

def create_hook_map(size):
    hook_map = [[0] * size for _ in range(size)]
    for i in range(size // 2 + 1):
        hook_id = size - 2 * i
        if hook_id < 1: continue
        for j in range(i, size - i):
            hook_map[i][j], hook_map[j][i] = hook_id, hook_id
    return hook_map

def create_hook_lookup(hook_map, size):
    lookup = {}
    for r in range(size):
        for c in range(size):
            hook_id = hook_map[r][c]
            if hook_id not in lookup: lookup[hook_id] = []
            lookup[hook_id].append((r, c))
    return lookup

def get_polyomino_shapes():
    return {"F":{(0,1),(1,0),(1,1),(1,2),(2,1)},"I":{(0,0),(0,1),(0,2),(0,3),(0,4)},"L":{(0,0),(1,0),(2,0),(3,0),(3,1)},"P":{(0,0),(0,1),(1,0),(1,1),(2,0)},"N":{(0,1),(0,2),(1,0),(1,1),(2,0)},"T":{(0,0),(0,1),(0,2),(1,1),(2,1)},"U":{(0,0),(0,2),(1,0),(1,1),(1,2)},"V":{(0,0),(1,0),(2,0),(2,1),(2,2)},"W":{(0,0),(1,0),(1,1),(2,1),(2,2)},"X":{(0,1),(1,0),(1,1),(1,2),(2,1)},"Y":{(0,1),(1,0),(1,1),(1,2),(1,3)},"Z":{(0,0),(0,1),(1,1),(2,1),(2,2)}}

def get_all_orientations(shape):
    orientations = set()
    current_shape = shape
    for _ in range(4):
        orientations.add(tuple(sorted(list(current_shape))))
        flipped_shape = frozenset({(r, -c) for r, c in current_shape})
        orientations.add(tuple(sorted(list(flipped_shape))))
        current_shape = frozenset({(-c, r) for r, c in current_shape})
    return [set(o) for o in orientations]

class DLXSolver:
    def __init__(self, columns, matrix):
        self.header = self._build_links(columns, matrix)

    def _build_links(self, columns, matrix):
        header = Node('header')
        col_nodes = {name: Node(name) for name in columns}
        for name in columns:
            col_node = col_nodes[name]
            header.link_right(col_node)

        for row_name, row_cols in matrix.items():
            first_node = None
            for col_name in row_cols:
                col_node = col_nodes[col_name]
                new_node = Node(row_name, col_node)
                col_node.u.link_down(new_node)
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
        self.name, self.col = name, col if col else self
        self.l, self.r, self.u, self.d = self, self, self, self
        self.size = 0 if col else 1

    def link_right(self, other):
        self.r, other.l = other, self

    def link_down(self, other):
        self.d, other.u = other, self
        self.col.size += 1

    def iter_right(self):
        curr = self.r
        while curr != self: yield curr; curr = curr.r

    def iter_left(self):
        curr = self.l
        while curr != self: yield curr; curr = curr.l

    def iter_down(self):
        curr = self.d
        while curr != self: yield curr; curr = curr.d

    def iter_up(self):
        curr = self.u
        while curr != self: yield curr; curr = curr.u

def can_be_tiled_dlx(board, required_pentominoes, size):
    target_cells = {(r, c) for r in range(size) for c in range(size) if board[r][c] != 0}

    if len(target_cells) != len(required_pentominoes) * 5:
        return None

    columns = [name for name, _ in required_pentominoes] + list(target_cells)
    matrix = {}
    row_num = 0

    for name, orientations in required_pentominoes:
        for shape in orientations:
            for r in range(size):
                for c in range(size):
                    placed_coords = {(r + ro, c + co) for ro, co in shape}
                    if placed_coords.issubset(target_cells):
                        matrix[row_num] = [name] + list(placed_coords)
                        row_num += 1

    if not matrix: return None

    solver = DLXSolver(columns, matrix)
    solution = solver.solve()

    if not solution: return None

    tiling_solution = {}
    for row in solution:
        name = next(item for item in row if isinstance(item, str))
        coords = {item for item in row if isinstance(item, tuple)}
        tiling_solution[name] = coords
    return tiling_solution

def check_pentomino_clues(tiling_solution, clues, size):
    if not tiling_solution: return False
    for r_idx, clue_list in enumerate(clues["rows"]):
        for clue in clue_list:
            if isinstance(clue, str):
                if clue not in tiling_solution: return False
                if not any(r == r_idx for r, c in tiling_solution[clue]): return False
    for c_idx, clue_list in enumerate(clues["cols"]):
        for clue in clue_list:
            if isinstance(clue, str):
                if clue not in tiling_solution: return False
                if not any(c == c_idx for r, c in tiling_solution[clue]): return False
    return True


def is_valid_number_placement(board, hook_lookup, clues, size):
    for hook_id in hook_lookup:
        if sum(1 for r,c in hook_lookup[hook_id] if board[r][c]==hook_id) != hook_id: return False
    for r_idx,clue_list in enumerate(clues["rows"]):
        for clue in clue_list:
            if isinstance(clue, int) and clue not in board[r_idx]: return False
    for c_idx,clue_list in enumerate(clues["cols"]):
        col_vals = [board[r][c_idx] for r in range(size)]
        for clue in clue_list:
            if isinstance(clue, int) and clue not in col_vals: return False
    return True


def find_best_empty_cell(board, hook_map, hook_lookup, size):
    best_cell, min_options = None, 11
    for r in range(size):
        for c in range(size):
            if board[r][c] == -1:
                options_count = 0
                for num in range(size+1): # Numbers from 0 to size
                    board[r][c] = num
                    count = sum(1 for row,col in hook_lookup[hook_map[r][c]] if board[row][col]==hook_map[r][c])
                    board[r][c] = -1
                    if count <= hook_map[r][c]: options_count += 1
                if options_count < min_options:
                    min_options, best_cell = options_count, (r, c)
                    if min_options == 0: return None
    return best_cell

def solve_numbers(board, hook_map, hook_lookup, required_pentominoes, clues, size):
    next_cell = find_best_empty_cell(board, hook_map, hook_lookup, size)

    if not next_cell:
        is_full = not any(-1 in row for row in board)
        if is_full and is_valid_number_placement(board, hook_lookup, clues, size):
            print("Found a valid number placement. Checking if it can be tiled with DLX...")
            tiling_solution = can_be_tiled_dlx(board, required_pentominoes, size)
            if tiling_solution:
                print("Found a valid tiling. Checking pentomino clues...")
                if check_pentomino_clues(tiling_solution, clues, size):
                    return True
        return False

    r, c = next_cell
    for num in range(size + 1): # Numbers from 0 to size
        board[r][c] = num
        count = sum(1 for row,col in hook_lookup[hook_map[r][c]] if board[row][col]==hook_map[r][c])
        if count <= hook_map[r][c]:
            if solve_numbers(board, hook_map, hook_lookup, required_pentominoes, clues, size):
                return True
    board[r][c] = -1
    return False

REQUIRED_PENTOMINO_NAMES = ["I", "U", "N", "X", "V", "Z"]
precondition_grid = [[0,0,0,0,4],[0,0,3,0,0],[3,2,0,5,0],[0,0,1,0,0],[0,0,0,0,1]]
outside_clues = {"rows":[["U"],["U"],[],["F"],["Y"]],"cols":[[],[],[],[],[]]}
grid_size = len(precondition_grid)

POLYOMINO_LIB = get_polyomino_shapes()
required_pentominoes = []
for name in REQUIRED_PENTOMINO_NAMES:
    if name not in POLYOMINO_LIB: raise ValueError(f"Shape '{name}' not found.")
    required_pentominoes.append((name, get_all_orientations(POLYOMINO_LIB[name])))

board = [[-1] * grid_size for _ in range(grid_size)]
for r in range(grid_size):
    for c in range(grid_size):
        if precondition_grid[r][c] != 0: board[r][c] = precondition_grid[r][c]

hook_map = create_hook_map(grid_size)
hook_lookup = create_hook_lookup(hook_map, grid_size)

print(f"Attempting to solve. Requires tiling {len(REQUIRED_PENTOMINO_NAMES)} pentominoes.")
if solve_numbers(board, hook_map, hook_lookup, required_pentominoes, outside_clues, grid_size):
    print("\nSolution found!")
    pprint.pprint(board)
else:
    print("\nNo solution was found")