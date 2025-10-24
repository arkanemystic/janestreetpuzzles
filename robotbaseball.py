from sympy import Symbol, simplify, lambdify
import numpy as np
from scipy.optimize import minimize_scalar
import time

p = Symbol('p', real=True, positive=True)
V = {}
Q = {}

V[(4, 0)] = 1
V[(4, 1)] = 1
V[(4, 2)] = 1
V[(0, 3)] = 0
V[(1, 3)] = 0
V[(2, 3)] = 0
V[(3, 3)] = 0

Q[(3, 2)] = 1
Q[(4, 0)] = 0
Q[(4, 1)] = 0
Q[(4, 2)] = 0
Q[(0, 3)] = 0
Q[(1, 3)] = 0
Q[(2, 3)] = 0
Q[(3, 3)] = 0

V[(3, 2)] = (4*p) / (1 + 4*p)
V[(2, 2)] = (4*p) / (2 + 4*p)
V[(1, 2)] = (4*p) / (3 + 4*p)
V[(0, 2)] = (4*p) / (4 + 4*p)

Q[(2, 2)] = ((1 + 4*p) / (2 + 4*p))**2
Q[(1, 2)] = ((1 + 4*p) / (3 + 4*p))**2
Q[(0, 2)] = ((1 + 4*p) / (4 + 4*p))**2

print("Starting backward iteration...")
start_time = time.time()

for s in [1, 0]:
    print(f"\nProcessing column s={s}")
    for b in [3, 2, 1, 0]:
        iter_start = time.time()
        print(f"  Computing state ({b}, {s})...")
        
        V_b_plus_1 = V[(b + 1, s)]
        V_s_plus_1 = V[(b, s + 1)]
        Q_b_plus_1 = Q[(b + 1, s)]
        Q_s_plus_1 = Q[(b, s + 1)]
        
        w = simplify(p * (4 - V_s_plus_1) / (V_b_plus_1 - V_s_plus_1 + p * (4 - V_s_plus_1)))
        
        V[(b, s)] = simplify(w * V_b_plus_1 + (1 - w) * V_s_plus_1)
        
        Q[(b, s)] = simplify(w**2 * Q_b_plus_1 + (1 - w**2 - (1 - w)**2 * p) * Q_s_plus_1)
        
        iter_time = time.time() - iter_start
        print(f"    V({b}, {s}) computed")
        print(f"    Q({b}, {s}) computed")
        print(f"    Elapsed: {iter_time:.1f}s")

grid_time = time.time() - start_time
print("\n" + "="*60)
print("Grid computation complete!")
print(f"Total grid filling time: {grid_time:.1f}s")
print("="*60)

q = Q[(0, 0)]
print("\nFinal q = Q(0, 0) obtained")

print("\nConverting symbolic expression to numerical function...")
conversion_start = time.time()
q_func = lambdify(p, q, 'numpy')
conversion_time = time.time() - conversion_start
print(f"Conversion completed in {conversion_time:.1f}s")

def neg_q(p_val):
    try:
        result = q_func(p_val)
        if isinstance(result, np.ndarray):
            result = float(result.item())
        else:
            result = float(result)
        return -result
    except:
        return np.inf

print("\nFinding maximum using numerical optimization...")
optimization_start = time.time()

result = minimize_scalar(neg_q, bounds=(0, 1), method='bounded', 
                        options={'xatol': 1e-12, 'maxiter': 500})

optimization_time = time.time() - optimization_start

p_optimal = result.x
q_maximal = -result.fun

print(f"Optimization completed in {optimization_time:.1f}s")

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(f"Optimal p: {p_optimal:.15f}")
print(f"Maximal q: {q_maximal:.10f}")
print("="*60)

print("\nVerification (checking nearby points):")
test_deltas = [-0.001, -0.0001, 0, 0.0001, 0.001]
for delta in test_deltas:
    p_test = p_optimal + delta
    if 0 <= p_test <= 1:
        q_test = float(q_func(p_test))
        marker = " <-- MAXIMUM" if delta == 0 else ""
        print(f"  p = {p_test:.6f}: q = {q_test:.10f}{marker}")

print("\nDerivative sign check:")
epsilon = 1e-8
q_left = float(q_func(p_optimal - epsilon))
q_center = float(q_func(p_optimal))
q_right = float(q_func(p_optimal + epsilon))

left_deriv = (q_center - q_left) / epsilon
right_deriv = (q_right - q_center) / epsilon

print(f"  Left derivative:  {left_deriv:.6e}")
print(f"  Right derivative: {right_deriv:.6e}")
if left_deriv > 0 and right_deriv < 0:
    print("  Confirmed: This is a local maximum")
elif abs(left_deriv) < 1e-6 and abs(right_deriv) < 1e-6:
    print("  Derivatives near zero - likely a critical point")
else:
    print("  Warning: Derivative behavior unexpected")

total_time = time.time() - start_time
print(f"\nTotal runtime: {total_time:.1f}s")
print("="*60)