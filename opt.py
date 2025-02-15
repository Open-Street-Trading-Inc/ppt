import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import time

# Black-Scholes Option Pricing Formula and Greeks
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    
    return delta, gamma, theta, vega, rho

# Streamlit App
st.title("Option Payoff and Greek Evolution")
st.write("Use the dropdown menu to switch between Option Payoff and Greeks. Each graph evolves over time.")

# Parameters
S_range = np.linspace(50, 150, 500)
K = 100
r = 0.05
sigma = 0.2
T_max = 1
time_snapshots = np.linspace(0.01, T_max, 50)

# Dropdown Menu
graph_type = st.selectbox(
    "Select Graph to Display:",
    ["Option Payoff", "Delta", "Gamma", "Theta", "Vega", "Rho"]
)

# Plotly Figure for the selected graph
progress_bar = st.progress(0)
placeholder = st.empty()

for i, T in enumerate(time_snapshots):
    values = []
    
    if graph_type == "Option Payoff":
        values = [black_scholes(S, K, T, r, sigma, option_type="call") for S in S_range]
        title = "Option Payoff Evolution"
        y_axis_title = "Payoff ($)"
    elif graph_type == "Delta":
        values = [calculate_greeks(S, K, T, r, sigma)[0] for S in S_range]
        title = "Delta Evolution"
        y_axis_title = "Delta"
    elif graph_type == "Gamma":
        values = [calculate_greeks(S, K, T, r, sigma)[1] for S in S_range]
        title = "Gamma Evolution"
        y_axis_title = "Gamma"
    elif graph_type == "Theta":
        values = [calculate_greeks(S, K, T, r, sigma)[2] for S in S_range]
        title = "Theta Evolution"
        y_axis_title = "Theta"
    elif graph_type == "Vega":
        values = [calculate_greeks(S, K, T, r, sigma)[3] for S in S_range]
        title = "Vega Evolution"
        y_axis_title = "Vega"
    elif graph_type == "Rho":
        values = [calculate_greeks(S, K, T, r, sigma)[4] for S in S_range]
        title = "Rho Evolution"
        y_axis_title = "Rho"
    
    # Plot the graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=values, mode='lines', name=graph_type))
    fig.update_layout(
        title=f"{title} at Time T={T:.2f}",
        xaxis_title="Stock Price ($)",
        yaxis_title=y_axis_title
    )
    
    # Update the graph
    placeholder.plotly_chart(fig, use_container_width=True)
    
    # Update the progress bar
    progress_bar.progress((i + 1) / len(time_snapshots))
    
    # Pause for animation effect
    time.sleep(0.1)
