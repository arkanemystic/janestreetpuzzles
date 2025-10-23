from sympy import Symbol, simplify, diff, solve, fraction, N
import time
start = time.time()
# Define symbolic variable
p = Symbol('p', real=True, positive=True)

# Initialize V and Q grids as dictionaries
# V(b, s): Expected score from state (b, s)
# Q(b, s): Probability of reaching state (3, 2) from state (b, s)
V = {}
Q = {}

# V(b, s) Boundary Conditions (Expected Score)
V[(4, 0)] = 1
V[(4, 1)] = 1
V[(4, 2)] = 1
V[(0, 3)] = 0
V[(1, 3)] = 0
V[(2, 3)] = 0
V[(3, 3)] = 0

# Q(b, s) Boundary Conditions (Probability of 3-2)
Q[(3, 2)] = 1
Q[(4, 0)] = 0
Q[(4, 1)] = 0
Q[(4, 2)] = 0
Q[(0, 3)] = 0
Q[(1, 3)] = 0
Q[(2, 3)] = 0
Q[(3, 3)] = 0

# Hand-Solved s=2 Column (Pre-fill these values)
V[(3, 2)] = (4*p) / (1 + 4*p)
V[(2, 2)] = (4*p) / (2 + 4*p)
V[(1, 2)] = (4*p) / (3 + 4*p)
V[(0, 2)] = (4*p) / (4 + 4*p)

Q[(2, 2)] = ((1 + 4*p) / (2 + 4*p))**2
Q[(1, 2)] = ((1 + 4*p) / (3 + 4*p))**2
Q[(0, 2)] = ((1 + 4*p) / (4 + 4*p))**2

print("Starting backward iteration...")

# Backward iteration: first s=1 column, then s=0 column
for s in [1, 0]:
    print(f"\nProcessing column s={s}")
    for b in [3, 2, 1, 0]:
        print(f"  Computing state ({b}, {s})...")
        
        # Get helper values
        V_b_plus_1 = V[(b + 1, s)]
        V_s_plus_1 = V[(b, s + 1)]
        Q_b_plus_1 = Q[(b + 1, s)]
        Q_s_plus_1 = Q[(b, s + 1)]
        
        # Calculate w (mixed strategy weight)
        w = simplify(p * (4 - V_s_plus_1) / (V_b_plus_1 - V_s_plus_1 + p * (4 - V_s_plus_1)))
        
        # Calculate V(b, s)
        V[(b, s)] = simplify(w * V_b_plus_1 + (1 - w) * V_s_plus_1)
        
        # Calculate Q(b, s)
        Q[(b, s)] = simplify(w**2 * Q_b_plus_1 + (1 - w**2 - (1 - w)**2 * p) * Q_s_plus_1)
        
        print(f"    V({b}, {s}) computed")
        print(f"    Q({b}, {s}) computed")
        print(f"Elapsed: {time.time() - start:.1f}s")

print("\n" + "="*60)
print("Grid computation complete!")
print("="*60)

# Get the final symbolic function q = Q(0, 0)
q = Q[(0, 0)]
print("\nFinal q = Q(0, 0) obtained")

# Find the derivative
print("\nComputing derivative...")
q_prime = diff(q, p)
print("Derivative computed")

# Find the numerator of q_prime
print("\nExtracting numerator of derivative...")
numerator, denominator = fraction(q_prime)
print("Numerator extracted")

# Solve for roots
print("\nSolving for critical points...")
solutions = solve(numerator, p)
print(f"Found {len(solutions)} solution(s)")

# Find the optimal p (between 0 and 1)
p_opt = None
for sol in solutions:
    # Check if solution is real and between 0 and 1
    sol_val = N(sol)
    if sol_val.is_real and 0 < sol_val < 1:
        p_opt = sol
        print(f"\nOptimal p found: {N(p_opt, 15)}")
        break

if p_opt is None:
    print("\nNo valid solution found in (0, 1)")
else:
    # Calculate the maximal q
    print("\nSubstituting optimal p into q...")
    q_max = q.subs(p, p_opt)
    q_max_simplified = simplify(q_max)
    
    # Convert to numerical value with high precision
    q_max_numerical = N(q_max_simplified, 15)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Optimal p: {N(p_opt, 15)}")
    print(f"Maximal q: {q_max_numerical:.10f}")
    print("="*60)
    print(f"Elapsed: {time.time() - start:.1f}s")

