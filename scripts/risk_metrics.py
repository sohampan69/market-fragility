import numpy as np
import pandas as pd


def calculate_percent_return(initial_capital, final_capital):
    return ((final_capital - initial_capital) / initial_capital) * 100


def calculate_sharpe_ratio(returns):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    return mean_return / std_dev if std_dev != 0 else 0


def calculate_sortino_ratio(returns):
    mean_return = np.mean(returns)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns)
    return mean_return / downside_std if downside_std != 0 else 0


def calculate_max_drawdown(equity_series):
    rolling_max = equity_series.cummax()
    drawdown = equity_series - rolling_max
    return drawdown.min()


def calculate_calmar_ratio(avg_return, max_drawdown):
    return avg_return / abs(max_drawdown) if max_drawdown != 0 else 0
