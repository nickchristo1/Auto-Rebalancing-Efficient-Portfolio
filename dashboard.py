# Nicholas Christophides  Nick.christophides@gmail.com

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from alpaca.trading.requests import GetPortfolioHistoryRequest

load_dotenv()

app = FastAPI()

initial_capital = 3900

# --- Alpaca Client ---
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(api_key, secret_key, paper=True)

# --- Optional static frontend ---
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/portfolio")
async def get_portfolio():
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()

    history_request = GetPortfolioHistoryRequest(
        period="1M",
        timeframe="1D"
    )
    history = trading_client.get_portfolio_history(history_request)

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

    return {
        "equity": equity,
        "pnl_daily": equity - last_equity,
        "daily_return_pct": daily_return_pct,
        "percent_return": percent_return,
        "positions": positions_data,
        "history": chart_data
    }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("dashboard:app", host="0.0.0.0", port=8000, reload=True)
