import numpy as np

# -----------------------------
# 1. Model and Tree Parameters
# -----------------------------
S0 = 100.0       # Initial stock price
K = 100.0        # Strike price
r = 0.05         # Risk-free rate (annual)
sigma = 0.20     # Volatility (annual)
T = 1.0          # Time to maturity (years)
n = 3            # Number of binomial steps

dt = T / n
u = np.exp(sigma * np.sqrt(dt))   # Up factor
d = np.exp(-sigma * np.sqrt(dt))  # Down factor
p = (np.exp(r*dt) - d) / (u - d)   # Risk-neutral probability
discount = np.exp(-r*dt)

# ----------------------------------
# 2. Build the Stock-Price Binomial
# ----------------------------------
stock_price = np.zeros((n+1, n+1))
for i in range(n+1):
    for j in range(i+1):
        stock_price[i, j] = S0 * (u**j) * (d**(i-j))

# ------------------------------
# 3. Compute Option Value (Call)
# ------------------------------
option_value = np.zeros((n+1, n+1))

# Final payoffs at maturity
for j in range(n+1):
    option_value[n, j] = max(stock_price[n, j] - K, 0)

# Backward induction
for i in range(n-1, -1, -1):
    for j in range(i+1):
        c_up   = option_value[i+1, j+1]
        c_down = option_value[i+1, j]
        option_value[i, j] = discount * (p * c_up + (1 - p) * c_down)

fair_value = option_value[0, 0]
print(f"Fair Value of the Call Option at time 0 = {fair_value:.4f}")
print("")

# ----------------------------------
# 4. Compute Hedge Portfolio (Delta)
# ----------------------------------
# Delta only makes sense at steps where there's a "next step."
# So delta[i,j] is for i = 0..(n-1), j = 0..i.
delta = np.zeros((n, n))
for i in range(n):
    for j in range(i+1):
        C_up   = option_value[i+1, j+1]
        C_down = option_value[i+1, j]
        S_up   = stock_price[i+1, j+1]
        S_down = stock_price[i+1, j]

        delta[i, j] = (C_up - C_down) / (S_up - S_down)

# Bond holding B_{i,j} = C_{i,j} - Delta_{i,j} * S_{i,j}
# We'll store B in same shape as option tree for convenience.
bond = np.zeros((n+1, n+1))
for i in range(n):
    for j in range(i+1):
        bond[i, j] = option_value[i, j] - delta[i, j] * stock_price[i, j]

# ---------------------------------------------
# 5. Print Stock Price, Option, and Hedge Data
# ---------------------------------------------
print("NODE-BY-NODE DATA (i = time step, j = number of up moves)")
print("---------------------------------------------------------")
for i in range(n+1):
    for j in range(i+1):
        S_ij = stock_price[i, j]
        C_ij = option_value[i, j]
        
        # For final layer (i=n), Delta & Bond are not used because there's no next step.
        if i < n:
            Delta_ij = delta[i, j]
            B_ij     = bond[i, j]
            hedging_portfolio_ij = Delta_ij * S_ij + B_ij * 1
        else:
            Delta_ij = 0.0
            B_ij     = 0.0
            hedging_portfolio_ij = 0
        
        print(f"Node (i={i}, j={j}): "
              f"Stock = {S_ij:.2f}, "
              f"Option = {C_ij:.2f}, "
              f"Delta = {Delta_ij:.4f}, "
              f"Bond = {B_ij:.4f}, "
              f"Hedging portfolio = {hedging_portfolio_ij:.4f}")
        
    print("")
