# Nicholas Christophides  Nick.christophides@gmail.com

import os
import math
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from alpaca.trading.requests import GetPortfolioHistoryRequest
import yfinance as yf
import pandas as pd
import numpy as np

tickers = ['AMT',  # American Tower      Sector: Real Estate
           'BRK-B',  # Berkshire Hathaway  Sector: Financials
           'CAT',  # Caterpillar         Sector: Industrials
           'COST',  # Costco              Sector: Consumer Staples
           'GE',  # GE Aerospace        Sector: Industrials
           'HD',  # Home Depot          Sector: Consumer Disc.
           'JNJ',  # Johnson & Johnson   Sector: Health Care
           'MSFT',  # Microsoft           Sector: Information Tech
           'NEE',  # NextEra Energy      Sector: Utilities
           'NVDA',  # NVIDIA              Sector: Information Tech
           'PG',  # Proctor & Gamble    Sector: Consumer Staples
           'PLD',  # Prologis            Sector: Real Estate
           'SPY',  # S&P 500             Market Index
           'TSLA',  # Tesla               Sector: Consumer Disc.
           'UNH',  # UnitedHealth        Sector: Health Care
           'V',  # Visa                Sector: Financials
           'XOM']  # ExxonMobil          Sector: Energy


# 1.) Set-up and Initialization
# -----------------------------

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


# 2.) Dashboard Backend
# ---------------------

@app.get("/api/portfolio")
async def get_portfolio():
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()

    history_request = GetPortfolioHistoryRequest(
        period="1M",
        timeframe="1D"
    )
    history = trading_client.get_portfolio_history(history_request)

    # # Calculate SPY vs. Strategy Returns
    # portfolio_equity = np.array(history.equity) if len(history.equity) > 1 else np.array([initial_capital])
    # portfolio_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]

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

                # Portfolio Weight
                "weight": float(p.market_value) / equity
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

    # Advanced Analytics: 1.) Find historical portfolio performance
    positions_dict = {  # Get a list of active positions
        p.symbol: p
        for p in positions
    }

    data = yf.download(tickers, period="1y", auto_adjust=True)["Close"]  # Extracted prices from yfinance
    data.index = pd.to_datetime(data.index)

    returns = data.pct_change(fill_method=None).dropna()  # Calculate returns of the active positions
    weights = []  # Get a list of active positions weights

    tickers[1] = "BRK.B"  # Fix name mismatch between Alpaca and yfinance
    for t in tickers:
        if t not in positions_dict:  # If no position held in asset, weight should be 0
            weights.append(0)
        else:
            weights.append(float(positions_dict[t].market_value) / equity)  # If held, find weight

    port_daily_rets = returns @ weights  # Daily returns for portfolio
    portfolio_growth = (1 + port_daily_rets).cumprod()  # Find compounded return
    strategy_return = portfolio_growth.iloc[-1] - 1  # Portfolio return over the year

    # Advanced Analytics: 2.) Calculate metrics
    # Drawdown
    running_max = portfolio_growth.cummax()
    drawdown = (portfolio_growth - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Volatility
    strategy_vol = port_daily_rets.std() * np.sqrt(252)
    spy_vol = np.std(spy_returns) if len(spy_returns) > 1 else 0.0

    # Return
    spy_return = (spy_prices[-1] / spy_prices[0]) - 1 if len(spy_prices) > 1 else 0.0
    alpha = strategy_return - spy_return

    # Sharpe Ratio
    mean_ret = np.mean(port_daily_rets)
    std_ret = np.std(port_daily_rets)
    std_ret = max(std_ret, 1e-6)
    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)

    # VaR
    var_99 = np.percentile(port_daily_rets, 1)

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
