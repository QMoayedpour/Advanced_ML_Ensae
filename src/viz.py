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


def plot_cumulatives(list_results, list_names):
    colors = plt.cm.tab10.colors
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(list_names)}

    fig, ax = plt.subplots(figsize=(12, 8))
    annotation_positions = []

    for result, model in zip(list_results, list_names):
        filtered = result.dropna().copy()
        filtered['cumulative_return'] = (1 + filtered['return_pf'] / 100).cumprod()
        filtered['return_pf'] = filtered['return_pf'] / 100

        returns_pf = filtered['return_pf']
        cumulative_return = filtered['cumulative_return']
        n = len(returns_pf)
        annual_return = (cumulative_return.iloc[-1]) ** (252 / n) - 1
        sharpe_ratio = (returns_pf.mean() / returns_pf.std()) * np.sqrt(252)

        index = filtered.index

        ax.plot(
            index,
            cumulative_return,
            label=f"{model}",
            color=model_colors.get(model, 'grey')
        )

        x_pos = index[-1]
        y_pos = cumulative_return.iloc[-1]

        offset_x = -500
        offset_y = 0

        for existing_x, existing_y in annotation_positions: # On veut éviter que 2 annotations soient confondues
            if abs(existing_y - y_pos) < 5:
                offset_y -= 5

        annotation_positions.append((x_pos, y_pos + offset_y))

        ax.annotate(
            f"Sharpe: {sharpe_ratio:.2f}\nAnnual Return: {annual_return:.2%}",
            xy=(x_pos, y_pos),
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color=model_colors.get(model, 'grey')),
            color=model_colors.get(model, 'grey'),
            fontsize=12,
            ha='left'
        )

    ax.set_title("", fontsize=16)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Cumulative Return", fontsize=14)
    ax.legend(title="Models", fontsize=12)
    ax.grid(False)

    plt.tight_layout()
    plt.show()
