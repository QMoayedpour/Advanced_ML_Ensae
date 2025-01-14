import numpy as np
import matplotlib.pyplot as plt


def plot_results(out, trainer):
    result = out.copy()

    prices = trainer.dataset.prices()

    plt.figure(figsize=(12,6))
    plt.plot(prices[prices.index.isin(result.dropna().index)], 
                            )
    plt.title("Price")
    plt.legend()
    plt.show()

    alloc_columns = [col for col in result.columns if col.endswith('_alloc')]
    result[alloc_columns].plot(figsize=(12,6), title="Allocations")
    plt.show()

    N = result.dropna().shape[0]
    log_returns_sum = np.sum(np.log(1+result['return_pf']/100))
    annual_rt = np.exp((252/N) * log_returns_sum) - 1
    sharpe = result['return_pf'].mean() / result['return_pf'].std() * np.sqrt(252)

    print(f"nb de jours d'investissement: {N}\nannualized return: {annual_rt}")
    print(f"sharpe ratio: {sharpe}")
    print(f"std deviation: {(result['return_pf']/100).std() * np.sqrt(252)}")
    print(f"downside_risk:  { ((result[ result['return_pf']<0 ]['return_pf']) / 100).std() * np.sqrt(252)} ")
