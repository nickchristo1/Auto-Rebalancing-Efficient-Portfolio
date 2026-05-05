# Nicholas Christophides  Nick.christophides@gmail.com

import os
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

app = FastAPI()

initial_capital = 3900

# --- Alpaca Client ---
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(api_key, secret_key, paper=True)
data_client = StockHistoricalDataClient(api_key, secret_key)

# --- SPY Data for comparison to strategy ---
spy_request = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame.Day,
    limit=30
)

spy_bars = data_client.get_stock_bars(spy_request).df
spy_bars = spy_bars.reset_index()

# --- Optional static frontend ---
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


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
    portfolio_equity = np.array(history.equity)
    portfolio_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]

    spy_prices = spy_bars["close"].values
    spy_returns = np.diff(spy_prices) / spy_prices[:-1]

    # Account information
    equity = float(account.equity)
    last_equity = float(account.last_equity)

    pnl_daily = equity - last_equity
    daily_return_pct = (pnl_daily / last_equity) * 100 if last_equity != 0 else 0
    percent_return = ((equity - initial_capital) / initial_capital) * 100

    positions_data = sorted(
        [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),

                # lifetime
                "unrealized_pl": float(p.unrealized_pl),
                "return_pct": float(p.unrealized_plpc) * 100,

                # today
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
    drawdown = (portfolio_equity - running_max) / running_max
    max_drawdown = drawdown.min()

    var_99 = np.percentile(portfolio_returns, 1)

    strategy_return = (portfolio_equity[-1] / portfolio_equity[0]) - 1
    spy_return = (spy_prices[-1] / spy_prices[0]) - 1
    alpha = strategy_return - spy_return

    strategy_vol = np.std(portfolio_returns)
    spy_vol = np.std(spy_returns)

    sharpe = (np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)) * np.sqrt(252)

    return {
        "equity": equity,
        "pnl_daily": equity - last_equity,
        "daily_return_pct": daily_return_pct,
        "percent_return": percent_return,
        "positions": positions_data,
        "history": chart_data,
        "analytics": {
            "max_drawdown": float(max_drawdown),
            "var_99": float(var_99),
            "strategy_return": float(strategy_return),
            "spy_return": float(spy_return),
            "alpha": float(alpha),
            "strategy_vol": float(strategy_vol),
            "spy_vol": float(spy_vol),
            "sharpe": float(sharpe)
        }
    }
