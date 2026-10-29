import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_cover_lines(ax, data, pcts, covers=(50, 75, 95), colors=('tab:blue', 'tab:purple', 'tab:red')):
    for p, cov, c in zip(pcts, covers, colors):
        v = np.nanpercentile(data, p)
        ax.axvline(v, ls='--', color=c, lw=1.2, label=f'{cov}% of trades: {v:.2f}')
    ax.legend()

def plot_MFE_MAE(groups):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex='row')
    # same bins for every column so the three histograms are directly comparable
    mfe_bins = np.histogram_bin_edges(groups['All']['MFE_ATR'].dropna(), bins=10)
    mae_bins = np.histogram_bin_edges(groups['All']['MAE_ATR'].dropna(), bins=10)

    for j, (name, d) in enumerate(groups.items()):
        ax1, ax2 = axes[0, j], axes[1, j]

        ax1.hist(d['MFE_ATR'].dropna(), bins=mfe_bins, density=True, facecolor='green', edgecolor='gray')
        ax1.set_xticks(mfe_bins, [f'{b:.1f}' for b in mfe_bins])
        ax1.set_xlabel('ATR'), ax1.set_ylabel('Density'), ax1.set_title(f'{name} MFE (ATR)')
        add_cover_lines(ax1, d['MFE_ATR'], pcts=(50, 75, 95))

        ax2.hist(d['MAE_ATR'].dropna(), bins=mae_bins, density=True, facecolor='orange', edgecolor='gray')
        ax2.set_xticks(mae_bins, [f'{b:.1f}' for b in mae_bins])
        ax2.set_xlabel('ATR'), ax2.set_ylabel('Density'), ax2.set_title(f'{name} MAE (ATR)')
        add_cover_lines(ax2, d['MAE_ATR'], pcts=(50, 25, 5))

    fig.tight_layout()


def scatter_ret_MFE_MAE(groups):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex='row', sharey='row')
    for j, (name, d) in enumerate(groups.items()):
        ax1, ax2 = axes[0, j], axes[1, j]

        ax1.scatter(d['MFE_ATR'], d['ret_ATR'], s=10)
        ax1.set_xlabel('MFE (ATR)'), ax1.set_ylabel('Return (ATR)'), ax1.set_title(f'{name}: MFE vs return')
        ax1.axhline(0, color='gray', lw=0.8, alpha=0.5, zorder=0)

        ax2.scatter(d['MAE_ATR'], d['ret_ATR'], s=10)
        ax2.set_xlabel('MAE (ATR)'), ax2.set_ylabel('Return (ATR)'), ax2.set_title(f'{name}: MAE vs return')
        ax2.axhline(0, color='gray', lw=0.8, alpha=0.5, zorder=0)

    fig.tight_layout()


def both_plots(trades):
    groups = {
    'All':     trades,
    'Winners': trades[trades['ret_ATR'] > 0],
    'Losers':  trades[trades['ret_ATR'] < 0]}   
    plot_MFE_MAE(groups)
    scatter_ret_MFE_MAE(groups)



def plot_mc_sim(trades,n_paths,length,start_bal):
    rng = np.random.default_rng(420)
    closed = trades[~trades['open']]
    rets = closed['ret%'].to_numpy() / 100

    samples = rng.choice(rets, size=(n_paths, length), replace=True)
    equity = start_bal * np.cumprod(1 + samples, axis=1)
    equity = np.hstack([np.full((n_paths, 1), start_bal), equity])   # start all paths at START

    mean, std = equity.mean(axis=0), equity.std(axis=0)
    med = np.median(equity, axis=0)
    x = np.arange(length + 1)

    final = equity[:, -1]
    p5, p95,p99 = np.percentile(final, [5, 95,99])
    p_loss = (final < start_bal).mean()

    cap = p95
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, equity.T, color='gray', alpha=0.01, lw=0.7)
    ax.plot(x, mean, color='k', lw=2, label=f'Mean (final {mean[-1]:.1f})')
    ax.plot(x, med, color='tab:red', lw=2, label=f'Median (final {med[-1]:.1f})')
    ax.fill_between(x, mean - std, mean + std, color='tab:blue', alpha=0.3,
                    label=f'±1 std (final {std[-1]:.1f})')
    ax.plot([], [], ' ', label=f'Final 5th/95th pct: {p5:.1f} / {p95:.1f}')
    ax.plot([], [], ' ', label=f'P(final < start): {p_loss:.1%}')
    ax.set_ylim(0, cap)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Equity')
    ax.legend(title=f'{n_paths:,} paths, {length} trades, start {start_bal}', loc='upper left')
    plt.show()

    final = equity[:, -1]
    p5, med, p95 = np.percentile(final, [5, 50, 95])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(final, bins=40, density=True, facecolor='steelblue', edgecolor='gray')
    ax.axvline(start_bal, color='k', lw=1.2, label=f'Start: {start_bal}')
    ax.axvline(med, color='tab:green', ls='--', lw=1.2, label=f'Median: {med:.1f}')
    ax.axvline(p5, color='tab:red', ls='--', lw=1.2, label=f'5th pct: {p5:.1f}')
    ax.axvline(p95, color='tab:purple', ls='--', lw=1.2, label=f'95th pct: {p95:.1f}')
    ax.set_xlabel('Final equity')
    ax.set_ylabel('Density')
    ax.set_xlim(0,p99)
    ax.set_title(f'Final equity distribution ({n_paths} paths, {length} trades)')
    ax.legend()
    plt.show()

    print(f'P(loss): {(final < start_bal).mean():.1%}')

    peak = np.maximum.accumulate(equity, axis=1)
    maxdd = (equity / peak - 1).min(axis=1) * 100          # % , negative

    d5, dmed, d95 = np.percentile(maxdd, [5, 50, 95])       # d5 = worst 5%

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(maxdd, bins=40, density=True, facecolor='indianred', edgecolor='gray')
    ax.axvline(dmed, color='tab:green', ls='--', lw=1.2, label=f'Median: {dmed:.1f}%')
    ax.axvline(d5, color='tab:red', ls='--', lw=1.2, label=f'Worst 5%: {d5:.1f}%')
    ax.axvline(d95, color='tab:purple', ls='--', lw=1.2, label=f'Mildest 5%: {d95:.1f}%')
    ax.set_xlabel('Max drawdown (%)')
    ax.set_ylabel('Density')
    ax.set_title(f'Bootstrap max drawdown distribution ({n_paths} paths, {length} trades)')
    ax.legend()
    plt.show()

    print(f'P(maxDD worse than -20%): {(maxdd < -20).mean():.1%}')

def dd_plot(eq, title=None):
    peak = eq.cummax()
    dd   = eq / peak - 1
    uw   = dd < 0

    last_peak = eq.index.to_series().where(~uw).ffill()
    tuw = (eq.index.to_series() - last_peak).dt.days

    ep  = (~uw).cumsum()
    dur = tuw.groupby(ep).max()
    dur = dur[dur > 0]

    print(f"MaxDD {dd.min():.1%} on {dd.idxmin():%Y-%m-%d} | "
          f"Longest underwater {tuw.max()}d | Median episode {dur.median():.0f}d | "
          f"Time underwater {uw.mean():.0%} | Currently {tuw.iloc[-1]}d under")

    fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    axs[0].plot(eq, color='k', lw=1.2, label='Equity')
    axs[0].plot(peak, color='tab:green', lw=0.8, ls='--', alpha=0.6, label='High-water mark')
    axs[0].set_ylabel('Equity')
    axs[0].legend(loc='upper left')

    axs[1].fill_between(dd.index, dd * 100, 0, color='indianred', alpha=0.6)
    axs[1].set_ylabel('Drawdown (%)')
    axs[1].set_title(f'Max DD {dd.min():.1%}', loc='left', fontsize=10)

    axs[2].fill_between(tuw.index, tuw, 0, color='steelblue', alpha=0.6)
    axs[2].set_ylabel('Days under water')
    axs[2].set_title(f'Longest {tuw.max()}d | median episode {dur.median():.0f}d | '
                     f'{uw.mean():.0%} of time underwater', loc='left', fontsize=10)
    if title is not None: 
        plt.suptitle(title)
        
    plt.show()


