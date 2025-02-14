import numpy as np
from dataclasses import dataclass

# ===========================================
#                 Functions
# ===========================================

def terminal_value_call(S_T, K):
    return max(0, S_T - K)

def terminal_value_put(S_T, K):
    return max(0, K - S_T)

@dataclass
class Node:
    S: float
    option_value: float = None

# ===========================================
#                 Parameters
# ===========================================

S_0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
steps = 1

# ===========================================
#                    Main
# ===========================================

dt = T / steps
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u
q_u = (np.exp(r * dt) - d) / (u - d)
q_d = 1 - q_u

# Forward pass, fill in stock prices
binary_tree = [
    [
        Node(
            S = S_0 * (u**(t-k)) * (d**(k)),
        )
     for k in range(t+1)] 
    for t in range(steps+1)
]

# Backward pass, fill in terminal option values
binary_tree[steps] = [
    Node(
        S = node.S,
        option_value = terminal_value_call(node.S, K)
    )
    for node in binary_tree[steps]
]

# Backward pass, fill in option values
for t in range(steps-1, -1, -1):
    for k in range(t+1):
        binary_tree[t][k].option_value = np.exp(-r * dt) * (
            q_u * binary_tree[t+1][k].option_value + 
            q_d * binary_tree[t+1][k+1].option_value
        )


print(f"u: {u}")
print(f"d: {d}")
# Print tree
for i in range(steps+1):
    for j in range(i+1):
        print(f"S: {binary_tree[i][j].S:.2f}, V: {binary_tree[i][j].option_value:.2f}", end=" | ")
    print()





