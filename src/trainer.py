from .dataset import FinDataset
import pandas as pd
import torch
import numpy as np
from tqdm import trange
import plotly.express as px


class Trainer(object):

    def __init__(self, model, tickers=["VTI", "AGG", "DBC", "^VIX"], device="cuda:0"):
        self.model = model
        self.dataset = FinDataset(tickers=tickers)
        self.tickers = tickers
        self.device = device
        self.result = None

        # Not custommable
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.2)
        self.scheduler_global = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)

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

        self.dataset.load_training_periods()

        for i in range(len(self.dataset.periods_train)):

            self.logs[i] = {}

            dataloader, X_test, periods = self.dataset.loader_period(i, rolling_window,
                                                                     batch_size, overlap,
                                                                     shuffle, verbose)
            
            loss_epochs = []

            for _ in trange(epochs) if verbose else range(epochs):

                loss_epochs.append(self._train_epoch(dataloader))

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

    def plot_results(self):

        if self.result is None:
            raise ValueError("Run the model first...")
        prices = self.dataset.prices()

        fig_prices = px.line(prices[prices.index.isin(self.result.dropna().index)], 
                             title="prices")
        fig_prices.show()

        alloc_columns = [col for col in self.result.columns if col.endswith('_alloc')]
        fig_allocations = px.line(self.result.dropna(), y=alloc_columns, 
                                  title="allocations")
        fig_allocations.show()

        N = self.result.dropna().shape[0]
        log_returns_sum = np.sum(np.log(1+self.result['return_pf']/100))
        annual_rt = np.exp((252/N) * log_returns_sum) - 1
        sharpe = self.result['return_pf'].mean() / self.result['return_pf'].std() * np.sqrt(252)

        print(f"nb de jours d'investissement: {N}\nannualized return: {annual_rt}")
        print(f"sharpe ratio: {sharpe}")
        print(f"std deviation: {(self.result['return_pf']/100).std() * np.sqrt(252)}")
        print(f"downside_risk:  { ((self.result[ self.result['return_pf']<0 ]['return_pf']) / 100).std() * np.sqrt(252)} ")
