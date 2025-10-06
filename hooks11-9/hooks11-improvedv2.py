# fixed_hooks11.py
from functools import reduce
from itertools import product
from operator import mul
from pprint import pprint

import numpy as np
from codetiming import Timer
from scipy.ndimage import label
# SciPy sum_labels fallback
try:
    from scipy.ndimage import sum_labels
except Exception:
    def sum_labels(arr, labels, index):
        return np.array([arr[labels == i].sum() for i in index])

from z3 import (
    And,
    Bool,
    BoolRef,
    Distinct,
    If,
    Implies,
    IntVector,
    IntNumRef,
    ModelRef,
    Not,
    Or,
    PbEq,
    Solver,
    sat,
    AtMost,
)

# ---------------------------
# Puzzle parameters (from user's original)
# ---------------------------
n = 9  # board size (9x9)

clues = np.array([
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 9, 0, 0, 0, 0],
])

edge_clues = {
    'top': {6: 7},  # Column 6 from top
    'right': {0: 'U', 3: 'X', 5: 2, 8: 'V'},  # Rows 0,3,5,8 from right
    'bottom': {2: 3},  # Column 2 from bottom
    'left': {0: 'I', 3: 6, 5: 'N', 8: 'Z'}  # Rows 0,3,5,8 from left
}

# create clues_dict mapping coords->value
clues_dict = {}
for i, j in product(range(n), repeat=2):
    if clues[i, j]:
        clues_dict[(i, j)] = int(clues[i, j])

# ---------------------------
# Utility helpers
# ---------------------------
def neighbours(i: int, j: int) -> list[tuple[int, int]]:
    """Return orthogonal neighbours inside the board."""
    direc = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neigh = [(i + di, j + dj) for di, dj in direc]
    return [(x, y) for x, y in neigh if 0 <= x < n and 0 <= y < n]


def connected(xm: np.ndarray) -> bool:
    """True iff the board's filled squares (non-zero) form a single connected component."""
    mask = (xm != 0).astype(int)
    if mask.sum() == 0:
        return False
    labels, num = label(mask)
    return num <= 1


def areas(xm: np.ndarray) -> int:
    """Product of the areas of the empty (unfilled) connected regions."""
    mat = np.where(xm == 0, 1, 0)
    labels, k = label(mat)
    if k == 0:
        return 0
    area = sum_labels(mat, labels, index=range(1, k + 1))
    return int(reduce(mul, area))


def evaluate_vars(m: ModelRef, vars_arr: np.ndarray) -> np.ndarray:
    """Evaluate z3 IntVector variables in model m and return numpy array of ints."""
    def ev(x):
        v = m.evaluate(x)
        # If z3 returns IntNumRef, convert to python int
        if isinstance(v, IntNumRef):
            return int(v.as_long())
        return int(v.as_long())
    vec = np.vectorize(ev)
    return vec(vars_arr)


# ---------------------------
# Pentomino shape utilities (canonical definitions)
# ---------------------------
PENTOMINOS = {
    'F': [(1, 0), (2, 0), (1, 1), (1, 2), (0, 1)],
    'I': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    'L': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)],
    'N': [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3)],
    'P': [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    'T': [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    'U': [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
    'V': [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    'W': [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    'X': [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
    'Y': [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    'Z': [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
}


def normalize_shape(coords):
    """Translate shape coords so minimum is (0,0) and return sorted list."""
    if not coords:
        return []
    coords = list(coords)
    min_r = min(r for r, c in coords)
    min_c = min(c for r, c in coords)
    return sorted(((r - min_r, c - min_c) for r, c in coords))


def rotate_90(coords):
    """Rotate coords 90 degrees clockwise (around origin)."""
    return [(c, -r) for r, c in coords]


def reflect_horizontal(coords):
    """Reflect coords horizontally (mirror over horizontal axis)."""
    return [(-r, c) for r, c in coords]


def get_all_orientations(coords):
    """Return list of unique orientations (rotations + reflections) normalized."""
    orientations = set()
    current = coords
    for _ in range(4):
        orientations.add(tuple(normalize_shape(current)))
        current = rotate_90(current)
    current = reflect_horizontal(coords)
    for _ in range(4):
        orientations.add(tuple(normalize_shape(current)))
        current = rotate_90(current)
    return [list(orient) for orient in orientations]


# ---------------------------
# Functions for pentomino verification (post-check)
# ---------------------------
def find_connected_components(board):
    """Find connected components of filled squares and return list of coords lists with length 5 only."""
    filled_mask = (board != 0).astype(int)
    labeled, num_components = label(filled_mask)
    components = []
    for i in range(1, num_components + 1):
        coords = list(zip(*np.where(labeled == i)))
        if len(coords) == 5:
            components.append(coords)
    return components


def shape_matches_pentomino(coords, pentomino_coords):
    """Check if coords match any orientation of pentomino_coords."""
    normalized_shape = normalize_shape(coords)
    all_orientations = get_all_orientations(pentomino_coords)
    normalized_orients = [normalize_shape(o) for o in all_orientations]
    return any(normalized_shape == o for o in normalized_orients)


def get_pentomino_sum(coords, board):
    """Sum of values in coords inside board."""
    return sum(int(board[r, c]) for r, c in coords)


# Edge clue helpers (updated to return both pent name and numeric value)
def get_first_seen_in_direction(board, assignments, row_or_col, index, direction):
    """
    Returns tuple (pentomino_name_or_None, numeric_value_or_None) for the first filled square seen.
    - row_or_col: 'row' or 'col'
    - index: row or column index
    - direction: 'forward' (left->right or top->bottom) or 'reverse'
    """
    if row_or_col == 'row':
        coords_to_check = [(index, j) for j in range(n)]
    else:
        coords_to_check = [(i, index) for i in range(n)]

    if direction == 'reverse':
        coords_to_check = coords_to_check[::-1]

    for r, c in coords_to_check:
        if board[r, c] != 0:
            # find pentomino containing (r,c) in assignments (if any)
            for coords_tuple, pent_name in assignments.items():
                if (r, c) in coords_tuple:
                    return pent_name, int(board[r, c])
            # no pent assignment (shouldn't happen if assignments cover all filled), still return numeric
            return None, int(board[r, c])
    return None, None


def verify_edge_clues(board, assignments):
    """Check that the edge clues match (letters refer to pentomino names, ints to numeric digits)."""
    for direction, clues_d in edge_clues.items():
        for index, expected in clues_d.items():
            if direction == 'top':
                pent, val = get_first_seen_in_direction(board, assignments, 'col', index, 'forward')
            elif direction == 'bottom':
                pent, val = get_first_seen_in_direction(board, assignments, 'col', index, 'reverse')
            elif direction == 'left':
                pent, val = get_first_seen_in_direction(board, assignments, 'row', index, 'forward')
            elif direction == 'right':
                pent, val = get_first_seen_in_direction(board, assignments, 'row', index, 'reverse')
            else:
                pent, val = None, None

            if isinstance(expected, int):
                if val != expected:
                    print(f"Edge clue mismatch (expected numeric): {direction} {index}, expected {expected}, got {val} (pent {pent})")
                    return False
            else:
                # expected is a letter like 'U' etc.
                if pent != expected:
                    print(f"Edge clue mismatch (expected pent): {direction} {index}, expected {expected}, got {pent}")
                    return False
    return True


def verify_pentomino_decomposition(board):
    """
    Verify that the filled squares can be decomposed into exactly 9 distinct pentominos,
    each sum multiple of 5, and edge clues are satisfied.
    Returns (is_valid, assignments, sums)
    assignments: dict mapping tuple(coords) -> pent name
    """
    components = find_connected_components(board)
    if len(components) != 9:
        print(f"Expected 9 pentominos (components of size 5), found {len(components)}")
        return False, {}, {}

    # check sums multiple of 5
    component_sums = []
    for coords in components:
        total = get_pentomino_sum(coords, board)
        if total % 5 != 0:
            print(f"Pentomino sum {total} is not a multiple of 5 for component {coords}")
            return False, {}, {}
        component_sums.append(total)

    used_pentominos = set()
    assignments = {}
    sums = {}

    for coords in components:
        found_match = False
        for name, shape in PENTOMINOS.items():
            if name in used_pentominos:
                continue
            if shape_matches_pentomino(coords, shape):
                used_pentominos.add(name)
                assignments[tuple(coords)] = name
                sums[name] = get_pentomino_sum(coords, board)
                found_match = True
                break
        if not found_match:
            print("Could not match a component to any unused pentomino; failed decomposition.")
            return False, {}, {}

    if len(used_pentominos) != 9:
        print(f"Expected 9 distinct pentominos, got {len(used_pentominos)}")
        return False, {}, {}

    # verify edge clues using assignments mapping
    if not verify_edge_clues(board, assignments):
        print("Edge clue verification failed after decomposition.")
        return False, {}, {}

    return True, assignments, sums


def print_pentomino_analysis(board, assignments, sums):
    print("\nPentomino Decomposition Analysis")
    print("=" * 50)
    for name in sorted(sums.keys()):
        print(f"{name}: Sum = {sums[name]}")
    print(f"\nTotal pentominos found: {len(assignments)}")
    print(f"All sums are multiples of 5: {all(s % 5 == 0 for s in sums.values())}")

    # visualization with letters
    vis_board = np.full(board.shape, ' ', dtype='U1')
    for coords_tuple, name in assignments.items():
        for r, c in coords_tuple:
            vis_board[r, c] = name
    print("\nPentomino assignment visualization:")
    for row in vis_board:
        print(' '.join(row))

    # edge verification trace
    print("\nEdge clue verification details:")
    for direction, clues_d in edge_clues.items():
        for index, expected in clues_d.items():
            if direction == 'top':
                pent, val = get_first_seen_in_direction(board, assignments, 'col', index, 'forward')
                print(f"Top of column {index}: expected {expected}, got pent={pent}, val={val}")
            elif direction == 'bottom':
                pent, val = get_first_seen_in_direction(board, assignments, 'col', index, 'reverse')
                print(f"Bottom of column {index}: expected {expected}, got pent={pent}, val={val}")
            elif direction == 'left':
                pent, val = get_first_seen_in_direction(board, assignments, 'row', index, 'forward')
                print(f"Left of row {index}: expected {expected}, got pent={pent}, val={val}")
            elif direction == 'right':
                pent, val = get_first_seen_in_direction(board, assignments, 'row', index, 'reverse')
                print(f"Right of row {index}: expected {expected}, got pent={pent}, val={val}")


# ---------------------------
# Hook placement enumeration
# ---------------------------
def generate_L_placements(n, s):
    """
    Generate all L-shaped placements (as lists of (r,c) coords) of size parameter s.
    Hook shape = horizontal leg of length s and vertical leg of length s meeting at a corner.
    The total number of cells in this hook = 2*s - 1 (corner counted once). For s from 1..n.
    """
    placements = []
    # Four orientation types for corner at (i0,j0) with leg lengths s:
    # 0: corner at top-left: horizontal right (j->j+s-1), vertical down (i->i+s-1)
    # 1: corner at top-right: horizontal left (j-s+1->j), vertical down
    # 2: corner at bottom-right: horizontal left, vertical up
    # 3: corner at bottom-left: horizontal right, vertical up
    for i0 in range(n):
        for j0 in range(n):
            # orientation 0: top-left corner
            if i0 + s <= n and j0 + s <= n:
                cells = set()
                for j in range(j0, j0 + s):
                    cells.add((i0, j))
                for i in range(i0, i0 + s):
                    cells.add((i, j0))
                placements.append(sorted(cells))
            # orientation 1: top-right corner
            if i0 + s <= n and j0 - s + 1 >= 0:
                cells = set()
                for j in range(j0 - s + 1, j0 + 1):
                    cells.add((i0, j))
                for i in range(i0, i0 + s):
                    cells.add((i, j0))
                placements.append(sorted(cells))
            # orientation 2: bottom-right corner
            if i0 - s + 1 >= 0 and j0 - s + 1 >= 0:
                cells = set()
                for j in range(j0 - s + 1, j0 + 1):
                    cells.add((i0, j))
                for i in range(i0 - s + 1, i0 + 1):
                    cells.add((i, j0))
                placements.append(sorted(cells))
            # orientation 3: bottom-left corner
            if i0 - s + 1 >= 0 and j0 + s <= n:
                cells = set()
                for j in range(j0, j0 + s):
                    cells.add((i0, j))
                for i in range(i0 - s + 1, i0 + 1):
                    cells.add((i, j0))
                placements.append(sorted(cells))
    # deduplicate placements
    unique = []
    seen = set()
    for p in placements:
        tup = tuple(sorted(p))
        if tup not in seen:
            seen.add(tup)
            unique.append(list(tup))
    return unique


# Precompute placements for each hook k (k=0..8), hook size s = n-k
placements_per_hook = []
for k in range(n):
    s_k = n - k
    placements = generate_L_placements(n, s_k)
    placements_per_hook.append(placements)
    # Debug sizes
    # print(f"Hook {k} size {s_k}: {len(placements)} placements (unique)")

# ---------------------------
# Z3 variables and constraints
# ---------------------------
# X: grid values as Ints (flattened into a vector then reshaped)
X = np.array(IntVector("x", n**2)).reshape((n, n))
# D: digit assigned to each hook (one per hook). Distinct 1..n
D = IntVector("d", n)

s = Solver()

# Domains
s.add(*[And(x >= 0, x <= n) for x in X.flat])
s.add(*[And(d >= 1, d <= n) for d in D])
s.add(Distinct(*D))

# The count constraint: there are exactly d cells equal to d on the board (for d in 1..n)
# (This is the same idea as your PbEq constraint)
s.add(*[PbEq([(x == d, 1) for x in X.flat], d) for d in range(1, n + 1)])

# Every 2x2 block must contain at least one zero
for i, j in product(range(n - 1), repeat=2):
    s.add(Or([X[i + di, j + dj] == 0 for di, dj in [(0, 0), (0, 1), (1, 0), (1, 1)]]))

# Optional: Require that filled squares have at least one filled neighbor (encourage connectivity)
for i, j in product(range(n), repeat=2):
    neigh = neighbours(i, j)
    if neigh:
        s.add(Implies(X[i, j] != 0, Or([X[a, b] != 0 for a, b in neigh])))

# Clue squares must be empty and neighbor sums equal clue
s.add(*[X[i, j] == 0 for i, j in clues_dict.keys()])
for (i, j), clue in clues_dict.items():
    s.add(sum(X[a, b] for a, b in neighbours(i, j)) == clue)

# Hook placement boolean variables H[k,p]
H = {}  # (k, p_index) -> Bool
for k in range(n):
    for p_idx, placement in enumerate(placements_per_hook[k]):
        H[(k, p_idx)] = Bool(f"H_{k}_{p_idx}")

# Exactly one placement chosen per hook
for k in range(n):
    s.add(Or([H[(k, p_idx)] for p_idx in range(len(placements_per_hook[k]))]))
    # ensure uniqueness: no two placements for the same hook simultaneously
    # (Or+NotAnd is sufficient; Z3 has Exactly but we use pairwise exclusion)
    for p1 in range(len(placements_per_hook[k])):
        for p2 in range(p1 + 1, len(placements_per_hook[k])):
            s.add(Not(And(H[(k, p1)], H[(k, p2)])))

# Disjointness of hooks: no board cell may be in more than one chosen placement
for i, j in product(range(n), repeat=2):
    # collect all placement booleans that include (i,j)
    membership_bools = []
    for k in range(n):
        for p_idx, placement in enumerate(placements_per_hook[k]):
            if (i, j) in placement:
                membership_bools.append(H[(k, p_idx)])
    if membership_bools:
        # At most one of those placements can be true (hooks disjoint)
        # Use AtMost (z3 builtin)
        s.add(AtMost(*membership_bools, 1))
        # Also: if a cell is non-zero, it must belong to some chosen placement (cover condition)
        s.add(Implies(X[i, j] != 0, Or(*membership_bools)))
    else:
        # No placement includes this cell (shouldn't happen), then force X[i,j]==0
        s.add(X[i, j] == 0)

# Link chosen placement to cell values: If H[k,p] is chosen, then for each (i,j) in that placement
# we must have X[i,j] == 0 OR X[i,j] == D[k]
for k in range(n):
    for p_idx, placement in enumerate(placements_per_hook[k]):
        b = H[(k, p_idx)]
        for (i, j) in placement:
            s.add(Implies(b, Or(X[i, j] == 0, X[i, j] == D[k])))

# Optional heuristic constraints you had earlier: last two hooks digits known (commented out)
# s.add(D[-1] == 1)
# s.add(D[-2] == 2)

# ---------------------------
# Solve with pentomino post-check (unchanged structure, but uses new placements)
# ---------------------------
def solve_with_pentomino_check(solver: Solver):
    while solver.check() == sat:
        m = solver.model()
        xm = evaluate_vars(m, X)
        # connectivity check
        if not connected(xm):
            print("Model found but filled region is not connected; excluding this model.")
            # exclude this solution
            solver.add(Not(And([X.flat[i] == int(xm.flat[i]) for i in range(n*n)])))
            continue

        # verify pentomino decomposition (post-check)
        is_valid, assignments, sums = verify_pentomino_decomposition(xm)
        if not is_valid:
            print("Model found but pentomino decomposition or edge clues failed; excluding model.")
            solver.add(Not(And([X.flat[i] == int(xm.flat[i]) for i in range(n*n)])))
            continue

        # If we reach here, we have a valid board
        print("\nValid board found!")
        print_pentomino_analysis(xm, assignments, sums)
        return xm

    return None


# ---------------------------
# Run solver
# ---------------------------
if __name__ == "__main__":
    with Timer():
        xm = solve_with_pentomino_check(s)

    if xm is not None:
        ans = areas(xm)
        print(f"\nFinal answer (product of empty region areas) = {ans}")
        print("Board (numeric grid):")
        pprint(xm)
    else:
        print("No valid solution found with pentomino decomposition.")
