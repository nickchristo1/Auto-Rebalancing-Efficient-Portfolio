# Nicholas Christophides  nick.christophides@gmail.com

""" In backtest.py the optimizer is back-tested over the last 6 months to determine how the portfolio it gives would
 perform over that time period.
 *NOTE: The optimizer will be given different data in the backtest than in the actual portfolio used for trading. As
 a result of this, this back-tester isn't meant to be used as proof of future performance of the portfolio that is
 given, but instead used as supporting evidence that the optimizer can produce portfolios that have approximately the
 desired level of return and volatility. """

from portfolio_optimization import eff_front_no_shorts, find_gmv_return
from estimate_cov_matrix import tickers, estimate_cov_matrix
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# 1.) Split the data into training and testing and run the optimization on the training
# -------------------------------------------------------------------------------------
data = yf.download(tickers, period="4y")["Close"].tail(700)  # Extracted prices from yfinance

# In-sample Set-up
in_sample_data = data.head(600).dropna()
in_sample_mu_hat = mu_hat = in_sample_data.mean().values
in_sample_log_returns = np.log(in_sample_data / in_sample_data.shift(1)).dropna()
in_sample_log_returns_std = (in_sample_log_returns - in_sample_log_returns.mean()) / in_sample_log_returns.std()
in_sample_cov_mat, _ = estimate_cov_matrix(np.cov(np.array(in_sample_log_returns_std), rowvar=False),
                                           in_sample_log_returns)

# Out-of-sample Set-up
out_of_sample_data = data.tail(100)

# In-Sample Optimization
mu_gmv = find_gmv_return(mu_hat, in_sample_cov_mat)
mus = np.linspace(mu_gmv, np.max(in_sample_mu_hat), 25)

optimal_weights = eff_front_no_shorts(mus[12], mu_hat, in_sample_cov_mat)


# 2.) Backtest the produced portfolio on the out-of-sample period
# ---------------------------------------------------------------
optimal_weights = np.array(optimal_weights)
daily_returns = out_of_sample_data.pct_change().dropna()
portfolio_daily_returns = daily_returns.dot(optimal_weights)
cumulative_growth = (1 + portfolio_daily_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_growth, label="Return over Time")
plt.xlabel("Time")
plt.ylabel("Return")
plt.title(f"Back-tested Portfolio over Last 100 Days (Out-of-Sample)\nReturn = {(cumulative_growth[-1] - 1)*100:.2f}%")
plt.grid(True)
plt.show()
