# Nicholas Christophides  Nick.christophides@gmail.com

import os
import numpy as np
import math
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf

# ----- 1.) Set-up and Initialization -----

load_dotenv()  # Load keys
app = FastAPI()

# --- Alpaca Client ---
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(api_key, secret_key, paper=False)

if os.path.isdir("static"):  # Static frontend
    app.mount("/static", StaticFiles(directory="static"), name="static")

initial_capital = 6285


def safe_float(x):
    if x is None:
        return 0.0
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return 0.0
    return float(x)


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


# ----- 2.) Dashboard Backend -----


@app.get("/api/portfolio")
async def get_portfolio():
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()

    history_request = GetPortfolioHistoryRequest(
        period="1M",
        timeframe="1D"
    )
    history = trading_client.get_portfolio_history(history_request)

    # Calculate SPY vs. Strategy Returns
    portfolio_equity = np.array(history.equity) if len(history.equity) > 1 else np.array([initial_capital])
    portfolio_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]

    spy = yf.download("SPY", period="1mo", interval="1d")["Close"].dropna()
    if spy is None or len(spy) < 2:
        spy_prices = np.array([1.0, 1.0])
        spy_returns = np.array([0.0])
    else:
        spy_prices = spy.to_numpy().flatten()
        spy_returns = np.diff(spy_prices) / np.where(spy_prices[:-1] == 0, 1e-8, spy_prices[:-1])

    # Account information
    equity = float(account.equity)  # Total assets summed
    last_equity = float(account.last_equity)  # Yesterday's final summed assets

    # PnL Metrics
    pnl_daily = equity - last_equity  # Today PnL
    daily_return_pct = (pnl_daily / last_equity) * 100 if last_equity != 0 else 0  # PnL in %
    cum_return = equity - initial_capital  # Cumulative return
    cum_percent_return = (cum_return / initial_capital) * 100  # Cumulative return in %

    positions_data = sorted(
        [
            {
                # Stock Info
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),

                # Lifetime PnL
                "unrealized_pl": float(p.unrealized_pl),
                "return_pct": float(p.unrealized_plpc) * 100,

                # Today PnL
                "today_pl": float(p.unrealized_intraday_pl)
                if hasattr(p, "unrealized_intraday_pl")
                else float(p.unrealized_pl),
                "today_return_pct": float(p.unrealized_intraday_plpc) * 100
                if hasattr(p, "unrealized_intraday_plpc")
                else float(p.unrealized_plpc) * 100,
            }
            for p in positions
        ],
        key=lambda x: x["unrealized_pl"],
        reverse=True
    )

    chart_data = [
        {
            "timestamp": history.timestamp[i],
            "equity": float(history.equity[i])
        }
        for i in range(len(history.equity))
    ]

    # Advanced Analytics
    running_max = np.maximum.accumulate(portfolio_equity)
    running_max = np.where(running_max == 0, 1e-8, running_max)

    drawdown = (portfolio_equity - running_max) / running_max
    max_drawdown = np.min(drawdown)

    strategy_vol = np.std(portfolio_returns) if len(portfolio_returns) > 1 else 0.0
    spy_vol = np.std(spy_returns) if len(spy_returns) > 1 else 0.0

    strategy_return = (portfolio_equity[-1] / portfolio_equity[0]) - 1 if len(portfolio_equity) > 1 else 0.0
    spy_return = (spy_prices[-1] / spy_prices[0]) - 1 if len(spy_prices) > 1 else 0.0

    alpha = strategy_return - spy_return

    mean_ret = np.mean(portfolio_returns) if len(portfolio_returns) > 1 else 0.0
    std_ret = np.std(portfolio_returns) if len(portfolio_returns) > 1 else 1e-8
    std_ret = max(std_ret, 1e-6)

    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)

    var_99 = np.percentile(portfolio_returns, 1) if len(portfolio_returns) > 1 else 0.0

    analytics = {
        "max_drawdown": safe_float(max_drawdown),
        "var_99": safe_float(var_99),
        "strategy_return": safe_float(strategy_return),
        "spy_return": safe_float(spy_return),
        "alpha": safe_float(alpha),
        "strategy_vol": safe_float(strategy_vol),
        "spy_vol": safe_float(spy_vol),
        "sharpe": safe_float(sharpe)
    }

    return {
        "equity": equity,
        "pnl_daily": equity - last_equity,
        "daily_return_pct": daily_return_pct,
        "cum_return": cum_return,
        "cum_percent_return": cum_percent_return,
        "positions": positions_data,
        "history": chart_data,
        "analytics": analytics
    }
