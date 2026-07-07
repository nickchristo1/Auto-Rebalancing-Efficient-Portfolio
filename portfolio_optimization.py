# Nicholas Christophides  nick.christophides@gmail.com

""" In portfolio_optimization.py the estimated covariance matrix from estimate_cov_matrix.py is employed in performing
portfolio optimization of the chosen assets.
The theoretically efficient portfolio is then used in auto_rebalance.py. """

import numpy as np
import pandas as pd
from estimate_cov_matrix import log_returns, pca_F
from scipy.optimize import minimize
from black_litterman import posterior_returns


# 1.) Find the Efficient Portfolio using the PCA Covariance Matrix Estimate and the Posterior Return Vector
# ----------------------------------------------------------------------------------------------------------

def eff_front_no_shorts(posterior_returns, cov_matrix, lmbda=3.0):
    """
    Uses Quadratic Optimization to minimize the variance of a portfolio for a target return level.
    :param lmbda: Risk aversion parameter
    :param posterior_returns: Optimal return vector from the Black-Litterman framework
    :param cov_matrix: covariance matrix
    :return: minimum variance no shorting allocations
    """
    n = len(posterior_returns)
    cov_annual = cov_matrix * 252

    # Objective Function: Minimize Portfolio Variance
    def objective(w):
        ret = w.T @ posterior_returns
        risk = w.T @ cov_annual @ w
        return -(ret - (lmbda * risk))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum of weights = 1)
    bounds = tuple((0, .15) for _ in range(n))
    init_guess = np.ones(n) / n

    res = minimize(objective, init_guess, method='SLSQP',
                   bounds=bounds, constraints=constraints)

    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")

    return res.x


# 2.) Choose an Expected Return and Find the Portfolio Weights
# ------------------------------------------------------------
optimal_weights = eff_front_no_shorts(posterior_returns, pca_F)

optimal_portfolio = pd.DataFrame({"Asset": log_returns.columns,
                                  "Weight": optimal_weights}
                                 ).sort_values(by="Weight", ascending=False).set_index("Asset")

print(f"Optimal Portfolio Weights: \n{optimal_portfolio.round(4)}\n"
      f"Total Portfolio Weight: {np.sum(optimal_weights):.4f}")
