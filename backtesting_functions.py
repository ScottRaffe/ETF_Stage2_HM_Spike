import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotting_functions as pf

def make_trade(d, i, x, is_open,cost_bps = 5):
    path = d['Close'][i:x + 1]
    p0, a0 = path[0], d['ATR14'][i]
    c = cost_bps / 1e4
    entry_px, exit_px = p0 * (1 + c), path[-1] * (1 - c)
    ret_g = 100 * (path[-1] / p0 - 1)          # gross
    ret   = 100 * (exit_px / entry_px - 1)     # net
    mfe   = 100 * (path.max() / p0 - 1)
    return {
        'Ticker': d['Ticker'][0],
        'entry_date': d['Date'][i], 'exit_date': d['Date'][x],
        'bars': x - i,
        'entry': entry_px, 'exit': exit_px,
        'ret%': ret,
        'ret_gross%': ret_g,
        'MFE%': mfe,
        'capture%': 100 * ret_g / mfe if mfe > 0 else np.nan,   # gross vs gross MFE
        'MAE%': 100 * (path.min() / p0 - 1),
        'ATR_entry': a0,
        'ATR_pct': 100 * a0 / p0,
        'ret_ATR': (exit_px - entry_px) / a0,                   # net
        'MFE_ATR': (path.max() - p0) / a0,
        'MAE_ATR': (path.min() - p0) / a0,
        'bars_to_MFE': int(path.argmax()),
        'bars_to_MAE': int(path.argmin()),
        'open': is_open,
    }

def trades_for(g, entry_fn, exit_fn,cost_bps):
    g = g.sort_values('Date').reset_index(drop=True)
    d = {c: g[c].to_numpy() for c in g.columns}   # column -> numpy array
    n, active, i, out = len(g), False, None, []

    for t in range(n):
        if active and exit_fn(d, t, i):            # exit checked first, never on entry bar
            out.append(make_trade(d, i, t, is_open=False,cost_bps=cost_bps))
            active = False
        if not active and entry_fn(d, t):          # can re-enter on the exit bar
            active, i = True, t

    if active:                                     # close any open position at end of data
        out.append(make_trade(d, i, n - 1, is_open=True,cost_bps=cost_bps))
    return out

def per_trade_backtester(df,entry_fn, exit_fn,cost_bsp):
    trades = pd.DataFrame([tr for _, g in df.groupby('Ticker') for tr in trades_for(g, entry_fn, exit_fn,cost_bsp)])
    summary = trades[~trades['open']].groupby('Ticker').agg(n=('ret%', 'size'),bars_med=('bars', 'median'),
        win=('ret%', lambda s: (s > 0).mean()),ATR_pct_med=('ATR_pct', 'median'),ret_per_mean=('ret%', 'mean'),
        MFE_per_mean=('MFE%', 'mean'),capture_med=('capture%', 'median'),MAE_per_mean=('MAE%', 'mean'),
        ret_ATR_mean=('ret_ATR', 'mean'),ret_ATR_med=('ret_ATR', 'median'),MFE_ATR_mean=('MFE_ATR', 'mean'),
        MAE_ATR_mean=('MAE_ATR', 'mean'),MFE_ATR_med=('MFE_ATR', 'median'),MAE_ATR_med=('MAE_ATR', 'median'))
    summary['MFE/|MAE| %']   = summary['MFE_per_mean'] / summary['MAE_per_mean'].abs()
    summary['MFE/|MAE| ATR'] = summary['MFE_ATR_mean'] / summary['MAE_ATR_mean'].abs()
    return trades,summary


def portfolio_backtester(trades, prices, capital=5000, risk=0.05, atr_mult=1,max_lev=5, whole=False, include_open=True):
    t = (trades if include_open else trades[~trades['open']]).reset_index(drop=True).copy()
    t['entry_date'] = pd.to_datetime(t['entry_date'])
    t['exit_date']  = pd.to_datetime(t['exit_date'])

    idx = prices.index[(prices.index >= t['entry_date'].min()) &
                       (prices.index <= t['exit_date'].max())]
    ent = t.groupby('entry_date').groups
    ext = t.groupby('exit_date').groups

    cash, held = capital, set()
    shares, eq_in, lev_in = (np.zeros(len(t)) for _ in range(3))
    tick, px_in = t['Ticker'].to_numpy(), t['entry'].to_numpy()
    eq = []

    for d in idx:
        for k in ext.get(d, []):                                   # exits first
            cash += shares[k] * t.at[k, 'exit']
            held.discard(k)
        val = {k: shares[k] * prices.at[d, tick[k]] for k in held}  # mark to close

        for k in ent.get(d, []):                                   # entries
            gross  = sum(val.values())
            equity = cash + gross
            n = equity * risk / (atr_mult * t.at[k, 'ATR_entry'])
            n = min(n, max(max_lev * equity - gross, 0) / px_in[k])
            if whole: 
                 n = np.floor(n)
            shares[k], eq_in[k] = n, equity
            val[k] = n * px_in[k]
            held.add(k)
            cash -= val[k]
            lev_in[k] = sum(val.values()) / equity

        eq.append(cash + sum(val.values()))

    t['shares'], t['eq_entry'], t['lev_entry'] = shares, eq_in, lev_in
    t['pnl'] = t['shares'] * (t['exit'] - t['entry'])
    t['R']   = t['pnl'] / (t['eq_entry'] * risk)
    return t, pd.Series(eq, index=idx)

def perf_stats(daily, t=None, rf=0.0, ppy=252):
    r   = daily.pct_change().dropna()
    ex  = r - rf / ppy
    yrs = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / yrs) - 1

    peak = daily.cummax()
    dd   = daily / peak - 1
    uw   = dd < 0
    ix   = daily.index.to_series()
    tuw  = (ix - ix.where(~uw).ffill()).dt.days

    trough = dd.idxmin()
    hi     = peak[trough]
    pre    = daily.loc[:trough]
    pk     = pre[pre >= hi].index[-1]                    # last day at the high before the trough
    post   = daily.loc[trough:]
    rec    = post[post >= hi].index
    rec    = rec[0] if len(rec) else None
    end    = rec if rec is not None else daily.index[-1]
    # print(f"Max DD: peak {pk:%Y-%m-%d} -> trough {trough:%Y-%m-%d} -> " + (f"recovered {rec:%Y-%m-%d}" if rec is not None else "NOT recovered"))

    out = {
        'Total return (%)':        100 * (daily.iloc[-1] / daily.iloc[0] - 1),
        'CAGR (%)':                100 * cagr,
        'Ann. vol (%)':            100 * r.std() * np.sqrt(ppy),
        'Sharpe':                  ex.mean() / ex.std() * np.sqrt(ppy),
        'Sortino':                 ex.mean() / np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(ppy),
        'Max DD (%)':              100 * dd.min(),
        'Calmar':                  cagr / abs(dd.min()),
        'Max DD duration (d)':     (end - pk).days,
        'Longest underwater (d)':  tuw.max(),
        'Time underwater (%)':     100 * uw.mean(),
    }

    if t is not None:
            f = t[t['shares'] > 0]                       # trades that actually got capital
            pnl = f['pnl']
            win, loss = pnl[pnl > 0], pnl[pnl < 0]
            n_open = (f.groupby('entry_date').size()
                        .sub(f.groupby('exit_date').size(), fill_value=0)
                        .reindex(daily.index, fill_value=0).cumsum())
            out.update({
                'Signals':                 len(t),
                'Trades filled':           len(f),
                'Skipped, no capacity (%)': 100 * (1 - len(f) / len(t)),
                'Win rate (%)':            100 * (pnl > 0).mean(),
                'Profit factor':           win.sum() / -loss.sum(),
                'Payoff (avg W / avg L)':  win.mean() / -loss.mean(),
                'Expectancy (R)':          f['R'].mean(),
                'Days in market (%)':      100 * (n_open > 0).mean(),
                'Peak leverage (x)':       f['lev_entry'].max(),
            })

    return pd.Series(out)

def combine(daily, spy_port, W_i, rebalance=True):
    """W_i = SPY weight, W_P = strategy weight. Weights not summing to 1 leave the remainder in cash at 0%."""
    spy_n = spy_port / spy_port.iloc[0]
    p_n   = daily / daily.iloc[0]
    W_i = min(W_i, 1)
    W_P = 1-W_i

    if rebalance:   # constant weights, rebalanced every day
        r  = W_i * spy_n.pct_change() + W_P * p_n.pct_change()
        eq = (1 + r.fillna(0)).cumprod()
    else:           # buy and hold, weights drift
        eq = W_i * spy_n + W_P * p_n

    return daily.iloc[0] * eq

def SPY_combo(daily,t,train_weight=0.0,is_train=True,title=None):
    spy = yf.download('SPY', start=daily.index.min(),
                  end=daily.index.max() + pd.Timedelta(days=1),
                  auto_adjust=True, progress=False)['Close'].squeeze()

    spy = spy.reindex(daily.index).ffill()      # force identical dates
    SPY_port = daily.iloc[0] * spy / spy.iloc[0]

    spy_stats = perf_stats(SPY_port)
    port_stats = perf_stats(daily, t)

    rho = np.corrcoef(daily.pct_change().dropna(), SPY_port.pct_change().dropna())[0,1]
    S_i,S_p = spy_stats['Sharpe'], port_stats['Sharpe']
    sigma_i, sigma_p = spy_stats['Ann. vol (%)'], port_stats['Ann. vol (%)']
    W_i = (sigma_p*(S_i - rho*S_p))/(sigma_i*(S_p - rho*S_i)+sigma_p*(S_i - rho*S_p))
    print(f'rho = {rho}')
    print(f'S_i = {S_i}')
    print(f'rho*S_p = {rho*S_p}')
    print(f'W_i = {W_i}')

    if is_train:
        mix = combine(daily, SPY_port, W_i=W_i)
    else:
        mix = combine(daily, SPY_port, W_i=train_weight)
        

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(daily, label = 'Strategy (2x)')
    ax.plot(SPY_port, label = 'SPY')
    ax.plot(mix, label = f'Mixed Portfolio: W_i={W_i}')
    # fig.suptitle('Full-Sample Performance')
    fig.tight_layout()
    fig.legend()

    stats = pd.DataFrame({
        'Strategy': perf_stats(daily, t),
        'SPY':      perf_stats(SPY_port),
        'Mix':      perf_stats(mix),
    })
    print(stats.round(2).to_string())
    pf.dd_plot(mix,title)

    if is_train:
        return W_i
    else:
        return None