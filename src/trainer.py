from src.dataset import FinDataset
import pandas as pd
import cvxpy as cp
import torch
import numpy as np
from tqdm import trange
import plotly.express as px

np.random.seed(702)

class Trainer(object):

    def __init__(self, model, tickers=["VTI", "AGG", "DBC", "^VIX"],
                 device=None, synthetic=False, lr=0.001, weight_decay=0.2, scheduler_gamma=0.8):
        
        self.scheduler_gamma = scheduler_gamma
        self.weight_decay = weight_decay
        self.model = model
        self.dataset = FinDataset(tickers=tickers, synthetic=synthetic)
        self.tickers = tickers
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.result = None

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        self.scheduler_global = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=scheduler_gamma)

    def _train_epoch(self, dataloader):
        loss = []
        for k, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_x, batch_y)
            loss.append(outputs.item())
            outputs.backward()
            self.optimizer.step()
        
        loss_epoch = sum(loss) / len(loss)
        return loss_epoch
    
    def train(self, epochs=200, initial_train_years=4, retrain_years=2, rolling_window=50, 
              shuffle=False, batch_size=64, overlap=True, verbose=False):

        self.logs = {}
        self.model.to(self.device)

        result = self.dataset.returns().copy()

        for col in result.columns:
            alloc_col = f"{col}_alloc"
            result[alloc_col] = np.nan

        self.dataset.load_training_periods(initial_train_years=initial_train_years, retrain_years=retrain_years)

        for i in range(len(self.dataset.periods_train)):
            
            self.logs[i] = {}

            dataloader, X_test, periods = self.dataset.loader_period(i, rolling_window,
                                                                     batch_size, overlap,
                                                                     shuffle, verbose)
            
            loss_epochs = []

            for _ in trange(epochs) if verbose else range(epochs):

                loss_epochs.append(self._train_epoch(dataloader))
                
            self.scheduler_global.step()

            with torch.no_grad():
                alloc_test = self.model.get_alloc_last(X_test.to(self.device)).cpu()

            original_data = self.dataset.returns().copy()

            filtered_data = original_data[(original_data.index >= periods[2]) & (original_data.index < periods[3])]

            alloc_columns = [f"{col}_alloc" for col in result.columns if not col.endswith("_alloc")]
            returns_columns = [col for col in result.columns if not col.endswith("_alloc")]

            result.loc[filtered_data.index, alloc_columns] = alloc_test.numpy()

            self.logs[i]["loss"] = loss_epochs

        rt_pf = 0

        for i in range(len(returns_columns)):
            rt_pf += result[alloc_columns[i]] * result[returns_columns[i]]
        result['return_pf'] = rt_pf

        self.result = result

        return result
    
    def markov_portfolio(self, initial_train_years=4, retrain_years=2, rolling_window=50, 
                        shuffle=False, batch_size=64, overlap=True, verbose=False):
        result = self.dataset.returns().copy()

        for col in result.columns:
            alloc_col = f"{col}_alloc"
            result[alloc_col] = np.nan

        self.dataset.load_training_periods()

        for i in range(len(self.dataset.periods_train)):

            _, X_test, periods = self.dataset.loader_period(i, rolling_window,
                                                            batch_size, overlap,
                                                            shuffle, verbose)

            original_data = self.dataset.returns().copy()

            train_start, train_end, test_start, test_end = periods

            for test_date in pd.date_range(test_start, test_end, freq='B'):

                train_data_end = test_date
                train_data_start = train_data_end - pd.Timedelta(days=rolling_window)

                estimation_data = original_data[(original_data.index < test_date)].tail(rolling_window)

                estimation_data = estimation_data.dropna()

                if len(estimation_data) > 0:

                    mu, Sigma = estimation_data.mean().to_numpy(), estimation_data.cov().to_numpy()

                    y_hat = self._compute_y_markowitz(mu, Sigma)

                    w = y_hat / y_hat.sum()

                    filtered_data = original_data[original_data.index == test_date]

                    alloc_columns = [f"{col}_alloc" for col in result.columns if not col.endswith("_alloc")]
                    returns_columns = [col for col in result.columns if not col.endswith("_alloc")]

                    result.loc[filtered_data.index, alloc_columns] = w

        rt_pf = 0

        for i in range(len(returns_columns)):
            rt_pf += result[alloc_columns[i]] * result[returns_columns[i]]
        result['return_pf'] = rt_pf

        return result

    def _compute_y_markowitz(self, mu, Sigma):

        y = cp.Variable(len(mu))

        objective = cp.Minimize(cp.quad_form(y, Sigma))

        constraints = [mu.T @ y == 1, y >= 0]

        problem = cp.Problem(objective, constraints)

        problem.solve()

        return y.value


    def plot_results(self, df=None):

        if df is None and self.result is None:
            raise ValueError("Run the model first...")
        
        elif df is None:
            result = self.result.copy()
        else:
            result = df.copy()

        prices = self.dataset.prices()

        fig_prices = px.line(prices[prices.index.isin(result.dropna().index)], 
                             title="prices")
        fig_prices.show()

        alloc_columns = [col for col in result.columns if col.endswith('_alloc')]
        fig_allocations = px.line(result.dropna(), y=alloc_columns, 
                                  title="allocations")
        fig_allocations.show()

        N = result.dropna().shape[0]
        log_returns_sum = np.sum(np.log(1+result['return_pf']/100))
        annual_rt = np.exp((252/N) * log_returns_sum) - 1
        sharpe = result['return_pf'].mean() / result['return_pf'].std() * np.sqrt(252)

        print(f"nb de jours d'investissement: {N}\nannualized return: {annual_rt}")
        print(f"sharpe ratio: {sharpe}")
        print(f"std deviation: {(result['return_pf']/100).std() * np.sqrt(252)}")
        print(f"downside_risk:  { ((result[ result['return_pf']<0 ]['return_pf']) / 100).std() * np.sqrt(252)} ")
