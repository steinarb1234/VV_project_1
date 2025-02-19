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
    S: float                   # Share price
    option_value: float = None # Intrinsic option value
    x: float = None            # Number of bonds  (in hedging portfolio)
    y: float = None            # Number of shares (in hedging portfolio)

# ===========================================
#                 Parameters
# ===========================================

S_0 = 100
K = 110
r = 0.035
sigma = 0.25
T = 1
N = 12
B_0 = 1

# ===========================================
#                    Main
# ===========================================

dt = T / N
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u
q_u = (np.exp(r * dt) - d) / (u - d)
q_d = 1 - q_u


# We build a binomial tree of nodes with stock prices,
# Then we calculate option values and hedging portfolio values.

# Forward pass, fill in stock prices
binomial_tree = [
    [
        Node(
            S = S_0 * (u**(t-k)) * (d**(k)),
        )
     for k in range(t+1)] 
    for t in range(N+1)
]

# Start of backward pass, fill in terminal option values
binomial_tree[N] = [
    Node(
        S = node.S,
        option_value = terminal_value_call(node.S, K)
    )
    for node in binomial_tree[N]
]

# Backward pass, fill in intrinsic option values at non-terminal nodes
for t in range(N-1, -1, -1):
    for k in range(t+1):
        binomial_tree[t][k].option_value = np.exp(-r * dt) * (
            q_u * binomial_tree[t+1][k].option_value + 
            q_d * binomial_tree[t+1][k+1].option_value
        )

# Forward pass, fill in hedging values
# TODO: Check N+1 or N
for t in range(0, N):
    for k in range(t+1):
        binomial_tree[t][k].y = (binomial_tree[t+1][k].option_value - binomial_tree[t+1][k+1].option_value) / (binomial_tree[t+1][k].S - binomial_tree[t+1][k+1].S)
        binomial_tree[t][k].x = (
            1/(1+r*dt) # TODO: Add B_0?????
            * (binomial_tree[t+1][k].option_value * binomial_tree[t+1][k+1].S - binomial_tree[t+1][k+1].option_value * binomial_tree[t+1][k].S)
            / (binomial_tree[t+1][k+1].S - binomial_tree[t+1][k].S)
        )


# Print the results:
# print(f"u: {u}")
# print(f"d: {d}")

# Print option values and stock prices
# for i in range(N+1):
for i in range(3):
    for j in range(i+1):
        print(f"S: {binomial_tree[i][j].S:.2f}, V: {binomial_tree[i][j].option_value:.2f}", end=" | ")
    print()

# Print hedging portfolio
# for i in range(N+1):
for i in range(3):
    for j in range(i+1):
        print(f"y: {binomial_tree[i][j].y:.2f}, x: {binomial_tree[i][j].x:.2f}, P: {binomial_tree[i][j].y * binomial_tree[i][j].S + binomial_tree[i][j].x * B_0 :.2f}", end=" | ")
    print()


# Convert the binomial tree to a graph and plot it
import matplotlib.pyplot as plt 
import networkx as nx

G = nx.Graph()
for i in range(N+1):
    for j in range(i+1):
        G.add_node(binomial_tree[i][j])

for i in range(N):
    for j in range(i+1):
        G.add_edge(binomial_tree[i][j], binomial_tree[i+1][j])
        G.add_edge(binomial_tree[i][j], binomial_tree[i+1][j+1])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()




