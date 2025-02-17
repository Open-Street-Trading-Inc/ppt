import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import time

# ---------------------------
# Black-Scholes pricing
# ---------------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Return the Black-Scholes price for a European call or put.
    S: Underlying price
    K: Strike
    T: Time-to-maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility
    option_type: "call" or "put"
    """
    # If T <= 0, option is expired -> intrinsic value
    if T <= 0:
        if option_type == "call":
            return np.maximum(S - K, 0.0)
        else:  # "put"
            return np.maximum(K - S, 0.0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ---------------------------
# Streamlit App
# ---------------------------
st.title("Option PnL Playground: Calls & Puts")
st.write(
    """
    This playground lets you build a portfolio of **calls and/or puts** (buy or sell),
    each with a maturity between **1 month** and **1 year**. The chart below displays 
    the **PnL** (profit or loss) over time rather than the raw option payoff.
    """
)

# Sidebar: Global parameters
st.sidebar.header("Global Parameters")
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)

st.sidebar.header("Underlying Price Range")
s_min = st.sidebar.number_input("Min Underlying Price", value=50)
s_max = st.sidebar.number_input("Max Underlying Price", value=150)
S_range = np.linspace(s_min, s_max, 500)

# Choose a "current" underlying price S0 at time 0 (for initial premium calculation)
S0 = st.sidebar.number_input("Current Underlying Price (S0) at t=0", value=100.0)

# ---------------------------
# Option positions
# ---------------------------
st.header("Define Option Positions")
st.write(
    """
    For each position, choose whether you are **buying** or **selling** a call or put option, 
    its strike, maturity (in months), and quantity.
    """
)

num_positions = st.number_input("Number of Option Positions", min_value=1, max_value=10, value=1, step=1)
positions = []

for i in range(int(num_positions)):
    st.subheader(f"Position {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        side = st.selectbox(f"Position {i+1} - Buy/Sell", ["Buy", "Sell"], key=f"side_{i}")
    with col2:
        # Now allow user to pick "Put" or "Call"
        option_type = st.selectbox(f"Position {i+1} - Option Type", ["Put", "Call"], key=f"type_{i}")
    strike = st.number_input(f"Position {i+1} - Strike Price", value=100.0, key=f"strike_{i}")
    maturity_months = st.slider(f"Position {i+1} - Maturity (months)", min_value=1, max_value=12, value=6, key=f"maturity_{i}")
    quantity = st.number_input(f"Position {i+1} - Quantity", min_value=1, value=1, step=1, key=f"qty_{i}")
    
    # Convert maturity from months to years
    maturity_years = maturity_months / 12.0
    positions.append({
        "side": side,               # "Buy" or "Sell"
        "type": option_type.lower(),# "call" or "put"
        "strike": strike,
        "maturity": maturity_years,
        "quantity": quantity
    })

# ---------------------------
# Calculate the initial cost (premium) of the portfolio
# ---------------------------
initial_cost = 0.0
for pos in positions:
    # Price this option at time 0 using S0
    cost = black_scholes(
        S0, 
        pos["strike"], 
        pos["maturity"], 
        r, 
        sigma, 
        option_type=pos["type"]
    )
    # If "Buy", you pay the premium (positive cost).
    # If "Sell", you receive the premium (negative cost).
    if pos["side"] == "Sell":
        cost = -cost
    
    # Multiply by quantity
    cost *= pos["quantity"]
    
    initial_cost += cost

st.write(
    f"**Initial Portfolio Cost (Premium paid if positive, received if negative):** "
    f"{initial_cost: .2f} $"
)

# ---------------------------
# PnL at time t
# ---------------------------
st.header("PnL Evolution")

# Current time slider (in years)
current_time = st.slider("Current Time (years)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Compute portfolio value at time t for each S in S_range
# Then subtract initial_cost to get PnL
portfolio_pnl = np.zeros_like(S_range)
for pos in positions:
    strike = pos["strike"]
    maturity = pos["maturity"]
    # time to maturity for this position
    ttm = maturity - current_time
    
    # Value of this option at time t for each S in S_range
    opt_val = black_scholes(
        S_range, strike, ttm, r, sigma, 
        option_type=pos["type"]
    )
    # If sold, the position value is negative the option value
    if pos["side"] == "Sell":
        opt_val = -opt_val
    opt_val *= pos["quantity"]
    
    portfolio_pnl += opt_val

# Convert from "portfolio value" to "PnL" by subtracting the initial cost
portfolio_pnl -= initial_cost

# Plot the PnL
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=S_range, 
    y=portfolio_pnl, 
    mode='lines', 
    name='Portfolio PnL'
))
fig.update_layout(
    title=f"Portfolio PnL at Time = {current_time:.2f} years",
    xaxis_title="Underlying Price ($)",
    yaxis_title="Portfolio PnL ($)"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Animation
# ---------------------------
if st.button("Animate PnL Evolution"):
    placeholder = st.empty()
    for t in np.linspace(0, 1, 50):
        port_val = np.zeros_like(S_range)
        for pos in positions:
            strike = pos["strike"]
            maturity = pos["maturity"]
            ttm = maturity - t
            opt_val = black_scholes(
                S_range, strike, ttm, r, sigma, 
                option_type=pos["type"]
            )
            if pos["side"] == "Sell":
                opt_val = -opt_val
            opt_val *= pos["quantity"]
            port_val += opt_val
        
        # PnL = current portfolio value - initial cost
        port_val -= initial_cost
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_range, 
            y=port_val, 
            mode='lines', 
            name='Portfolio PnL'
        ))
        fig.update_layout(
            title=f"Portfolio PnL at Time = {t:.2f} years",
            xaxis_title="Underlying Price ($)",
            yaxis_title="PnL ($)"
        )
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.1)

