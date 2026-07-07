# Nicholas Christophides  nick.christophides@gmail.com

"""
In ridge.py, ridge regression is used in order to make weekly estimates on return magnitudes of stocks by using a
walk forward approach. These return estimates are used as information for the black-litterman return vector approach,
that will allow the regressions to be input to the return vector, to provide more informative data than simply using
historical return.
"""

import pandas as pd
import numpy as np
from estimate_cov_matrix import prices
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr


def prepare_panel_data(daily_prices):
    """
    Transforms daily price data into a weekly cross-sectional panel with features and targets.
    """
    # 1. Resample to Weekly (using Friday closes)
    weekly_prices = daily_prices.resample('W-FRI').last()

    # 2. Calculate Forward 1-Week Log Return
    # Shift by -1 so that the features calculated this Friday align with the return next Friday
    forward_returns = np.log(weekly_prices / weekly_prices.shift(1)).shift(-1)

    # 3. Calculate Price-Based Features
    mom_1m = np.log(weekly_prices / weekly_prices.shift(4))  # 1 month
    mom_6m = np.log(weekly_prices / weekly_prices.shift(26))  # 6 months
    mom_12m = np.log(weekly_prices / weekly_prices.shift(52))  # 1 Year

    # Volatility (Using daily data, then resampling to weekly)
    daily_returns = np.log(daily_prices / daily_prices.shift(1))
    vol_3m = daily_returns.rolling(window=63).std() * np.sqrt(252)  # 63 trading days ~ 3 months, annualized
    vol_3m_weekly = vol_3m.resample('W-FRI').last()

    vol_1m = daily_returns.rolling(window=22).std() * np.sqrt(252)  # 22 trading days ~ 1 months, annualized
    vol_1m_weekly = vol_1m.resample('W-FRI').last()

    # 4. Structure the Panel Data
    # Stack the dataframes to create a long format panel: MultiIndex (Date, Ticker)
    panel = pd.DataFrame({
        'Target_Fwd_Ret': forward_returns.stack(),
        'Mom_1M': mom_1m.stack(),
        'Mom_6M': mom_6m.stack(),
        'Mom_12M': mom_12m.stack(),
        'Vol_3M': vol_3m_weekly.stack(),
        'Vol_1M': vol_1m_weekly.stack()
    })

    panel = panel.dropna()

    return panel


def cross_sectional_standardize(panel_df, feature_cols):
    """
    Applies cross-sectional z-scoring to the features for each period.
    """
    standardized_panel = panel_df.copy()

    # Group by Date (level=0) and standardize across the assets for each feature
    standardized_panel[feature_cols] = panel_df.groupby(level=0)[feature_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    return standardized_panel


def walk_forward_ridge(panel_df, feature_cols, target_col='Target_Fwd_Ret', alpha=1.0, min_train_weeks=52):
    """
    Performs expanding-window walk-forward prediction over the panel data.
    """
    df = panel_df.sort_index(level=0)
    dates = df.index.get_level_values(0).unique()

    results = []

    # Iterate through time, starting after our minimum training window
    for i in range(min_train_weeks, len(dates) - 1):
        train_dates = dates[:i]
        test_date = dates[i]

        # Safely slice the MultiIndex panel
        idx = pd.IndexSlice
        train_data = df.loc[idx[train_dates, :], :]
        test_data = df.loc[idx[test_date, :], :]

        X_train = train_data[feature_cols]
        y_train = train_data[target_col]

        X_test = test_data[feature_cols]

        # Fit Ridge model
        model = Ridge(alpha=alpha, solver='svd')
        model.fit(X_train, y_train)

        # Predict out-of-sample for the test week
        preds = model.predict(X_test)

        # Store results
        test_res = test_data.copy()
        test_res['Predicted_Ret'] = preds
        results.append(test_res[[target_col, 'Predicted_Ret']])

    # Concatenate all out-of-sample predictions
    oos_results = pd.concat(results)
    return oos_results


def evaluate_predictions(oos_results, target_col='Target_Fwd_Ret', pred_col='Predicted_Ret'):
    """
    Evaluates predictions using Cross-Sectional Information Coefficient (Rank IC).
    """

    def calc_ic(group):
        corr, _ = spearmanr(group[target_col], group[pred_col])
        return corr

    # Calculate IC for each week
    ic_series = oos_results.groupby(level=0).apply(calc_ic).dropna()

    mean_ic = ic_series.mean()

    # Information Ratio of the IC (Mean IC / Std Dev of IC)
    ic_ir = mean_ic / ic_series.std() if ic_series.std() != 0 else 0

    return mean_ic, ic_ir


def optimize_ridge_penalty(panel_df, feature_cols, target_col='Target_Fwd_Ret',
                           alphas=np.logspace(-2, 4, 15), min_train_weeks=52):
    """
    Tests a grid of L2 penalties to find the optimal alpha for the dataset.
    """
    print(f"{'Alpha':>10} | {'Mean IC':>10} | {'IC IR':>10}")
    print("-" * 37)

    best_alpha = None
    best_ir = -np.inf
    best_oos_results = None

    for alpha in alphas:
        oos_res = walk_forward_ridge(panel_df, feature_cols, target_col, alpha, min_train_weeks)
        mean_ic, ic_ir = evaluate_predictions(oos_res, target_col)

        print(f"{alpha:10.2f} | {mean_ic:10.4f} | {ic_ir:10.4f}")

        # Optimize for IC IR here, could optimize for MSE or Mean IC
        if ic_ir > best_ir:
            best_ir = ic_ir
            best_alpha = alpha
            best_oos_results = oos_res

    print("-" * 37)
    print(f"Optimal Alpha Selected: {best_alpha:.2f}")

    return best_alpha, best_oos_results


def apply_ema_smoothing(oos_results, pred_col='Predicted_Ret', span=3):
    """
    Applies an Exponential Moving Average to the predictions for each ticker over time.
    A span of 3 or 4 weeks is standard for mid-frequency weekly models.
    """
    smoothed_results = oos_results.copy()

    # Group by Ticker (level=1 in a MultiIndex of [Date, Ticker]), apply the EMA to the prediction column
    smoothed_preds = smoothed_results.groupby(level=1)[pred_col].transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )

    smoothed_results['Smoothed_Predicted_Ret'] = smoothed_preds

    return smoothed_results.dropna(subset=['Smoothed_Predicted_Ret'])


# --- Execution ---
panel_data = prepare_panel_data(prices)

# Standardize the features cross-sectionally
features = ['Mom_1M', 'Mom_6M', 'Mom_12M', 'Vol_3M']
standardized_panel = cross_sectional_standardize(panel_data, features)
alphas_to_test = np.logspace(0, 4, 10)  # Testing penalties from 1 to 10,000
optimal_alpha, raw_predictions = optimize_ridge_penalty(standardized_panel, features,
                                                        alphas=alphas_to_test, min_train_weeks=20)

# Apply the 4-week EMA to the raw predictions (Exponential Moving Average)
smoothed_predictions = apply_ema_smoothing(raw_predictions, span=4)
