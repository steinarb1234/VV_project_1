
# Multi-step hedging portfolio binary tree model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def call_option_terminal(S_T, K):
    return max(0, S_T - K)

def put_option_terminal(S_T, K):
    return max(0, K - S_T)

def binomial_tree(S, K, r, T, sigma, N, terminal_payoff):
    """
    Function to calculate the option price using the binomial tree model
    :param S: Initial stock price
    :param K: Strike price
    :param r: Risk-free rate
    :param T: Time to maturity
    :param sigma: Volatility
    :param N: Number of steps
    :param option_type: Option type (call or put)
    :return: Option price
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    q = 1 - p

    # Initialize the stock price tree
    stock_price = np.zeros((N + 1, N + 1))
    stock_price[0, 0] = S
    for i in range(1, N + 1):
        stock_price[i, 0] = stock_price[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_price[i, j] = stock_price[i - 1, j - 1] * d

    # Initialize the option price tree
    option_price = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        option_price[N, j] = terminal_payoff(stock_price[N, j], K)

    # Calculate the option price at time t=0
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_price[i, j] = np.exp(-r * dt) * (p * option_price[i + 1, j] + q * option_price[i + 1, j + 1])

    return option_price[0, 0]


if __name__ == "__main__":
    # Option parameters
    S = 100  # Initial stock price
    K = 100  # Strike price
    r = 0.05  # Risk-free rate
    T = 1  # Time to maturity
    sigma = 0.2  # Volatility
    N = 100  # Number of steps
    terminal_payoff = call_option_terminal
    # terminal_payoff = put_option_terminal

    # Calculate the option price using the binomial tree model
    option_price = binomial_tree(S, K, r, T, sigma, N, terminal_payoff)
    print(f"Option price: {option_price}")

    # Plot the option price as a function of the number of steps
    N_values = np.arange(10, 1000, 10)
    option_prices = [binomial_tree(S, K, r, T, sigma, N, terminal_payoff) for N in N_values]

    plt.figure(figsize=(10, 5))
    plt.plot(N_values, option_prices, label="Option Price")
    plt.xlabel("Number of Steps")
    plt.ylabel("Option Price")
    plt.title("Option Price as a Function of the Number of Steps")
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate the Black-Scholes option price
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    print(f"Black-Scholes Call Option Price: {bs_call_price}")

    # Calculate the error between the binomial tree model and the Black-Scholes model
    errors = [abs(option_price - bs_call_price) for option_price in option_prices]

    plt.figure(figsize=(10, 5))
    plt.plot(N_values, errors, label="Error")
    plt.xlabel("Number of Steps")
    plt.ylabel("Error")
    plt.title("Error between Binomial Tree and Black-Scholes Models")
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate the convergence rate
    log_errors = np.log(errors)
    log_N_values = np.log(N_values)
    slope, intercept = np.polyfit(log_N_values, log_errors, 1)
    print(f"Convergence rate: {slope}")

    # Plot the convergence rate
    plt.figure(figsize=(10, 5))
    plt.plot(log_N_values, log_errors, label="Log Error")
    plt.plot(log_N_values, slope * log_N_values + intercept, label="Fitted Line")
    plt.xlabel("Log Number of Steps")
    plt.ylabel("Log Error")
    plt.title("Convergence Rate of the Binomial Tree Model")
    plt.legend()
    plt.grid()
    plt.show()













