import numpy as np
from dataclasses import dataclass

def terminal_value_call(S_T, K):
    return max(0, S_T - K)

def terminal_value_put(S_T, K):
    return max(0, K - S_T)

@dataclass
class Node:
    S: float
    option_value: float


# class Node:
#     def __init__(self, S, t, terminal_value):
#         if t == T: # Terminal node
#             self.up = None
#             self.down = None
#             self.option_value = terminal_value(S, K)
#         else:
#             self.up = Node(S * u, t + dt, terminal_value)
#             self.down = Node(S * d, t + dt, terminal_value)
#             self.option_value = np.exp(-r * dt) * (q_u * self.up.option_value + q_d * self.down.option_value)

#         self.S = S
#         self.t = t

#     def y(self):
#         pass

S_0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
steps = 1

dt = T / steps
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u
q_u = (np.exp(r * dt) - d) / (u - d)
q_d = 1 - q_u


binary_tree = [
    [
        Node(
            S = S_0 * (u**t) * (d**(k-t)),
            option_value = 0,
        )
     for k in range(t+1)] 
    for t in range(steps+1)
]

print(u, d)
print(binary_tree)






