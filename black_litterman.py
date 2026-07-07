# Nicholas Christophides  nick.christophides@gmail.com

"""
The Black-Litterman approach is used here to find a better representation of the expected return vector than the
historical mean-return vector. It uses the ridge regression estimates as input to allow for a hybrid between
forecasting and theoretical market efficiency.
"""

import yfinance as yf
import pandas as pd
from estimate_cov_matrix import tickers, pca_F
from ridge import smoothed_predictions
import numpy as np
from datetime import datetime


# 1.) Find Market Caps
market_caps = {}
for ticker_symbol in tickers:
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        cap = ticker_obj.info.get('marketCap')
        market_caps[ticker_symbol] = cap
    except Exception as e:
        print(f"Could not fetch cap for {ticker_symbol}: {e}")
        market_caps[ticker_symbol] = None

caps = pd.Series(market_caps)
caps = caps.fillna(caps.median())
market_weights = caps / caps.sum()

# 2.) Calculate Market Equilibrium (PI)
delta = 3.0  # Standard
cov_matrix = pd.DataFrame(pca_F, index=tickers, columns=tickers)
market_weights = (caps / caps.sum()).loc[tickers]

pi = delta * cov_matrix.dot(market_weights)
pi_annual = pi * 252

pi_series = pd.Series(pi_annual, index=tickers)

# 3.) Construct the Black-Litterman Return Vector
today_str = datetime.now().strftime('%Y-%m-%d')

if today_str in smoothed_predictions.index.get_level_values(0):
    target_date = today_str
else:
    # If today isn't a trading day, grab the most recent available date
    target_date = smoothed_predictions.index.get_level_values(0).max()

tau = 0.05
P = np.eye(len(tickers))  # P is an identity matrix because we have a view on every asset
Q = smoothed_predictions.xs(target_date, level=0)['Smoothed_Predicted_Ret'] * 52
Q = Q.clip(lower=-0.5, upper=1)  # Don't allow for overly extreme predictions

# Omega: The uncertainty of the views
omega = np.diag(np.diag(tau * cov_matrix))

# Calculate the Posterior Expected Return (E[R]) (blended return expectation)
inv_tau_sigma = np.linalg.inv(tau * cov_matrix)
term1 = np.linalg.inv(inv_tau_sigma + P.T @ np.linalg.inv(omega) @ P)
term2 = (inv_tau_sigma @ pi) + (P.T @ np.linalg.inv(omega) @ Q)
posterior_returns = term1 @ term2

# View Return Vector
final_er_series = pd.Series(posterior_returns, index=tickers)

comparison = pd.DataFrame({
    'Equilibrium_Pi': pi_series,
    'Posterior_ER': final_er_series
}, index=tickers)

print(comparison)
