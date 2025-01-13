import yfinance as yf
import pandas as pd
from datetime import timedelta
import torch
from torch.utils.data import DataLoader, TensorDataset


class FinDataset:
    def __init__(self, tickers=['VTI', 'AGG', 'DBC', '^VIX'], start_date="2006-03-01", end_date="2020-12-31"):
        """
        Initialise le dataset avec les tickers et les dates de début et de fin
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._get_data_yfinance()
    
    def _get_data_yfinance(self):
        """
        Télécharge les données financières de Yahoo Finance et les prépare (prix, rendements, etc.)
        """
        data = {}
        prices = yf.download(self.tickers, start=self.start_date, end=self.end_date, interval="1d")['Close']
        prices.index = prices.index.tz_localize(None).floor('D')
        data["prices"] = prices
        returns = prices.pct_change() * 100
        data["returns"] = returns
        returns_prices = pd.concat([prices.shift(1), returns], axis=1, ignore_index=True)
        returns_prices.columns = [f'{col}_St-1' for col in prices.columns] + [f'{col}_rt' for col in prices.columns]
        data["return_prices"] = returns_prices.dropna()
        return data

    def _generate_training_periods(self, initial_train_years=4, retrain_years=2):
        """
        Génère les périodes d'entraînement et de test en fonction des années d'entraînement initial et de réentraînement
        """
        training_periods = []
        test_periods = []

        last_date_1st_training = self.data["returns"].index[initial_train_years * 252]
        training_periods.append((self.data["returns"].index[0], last_date_1st_training))

        first_invest_date = last_date_1st_training
        end_date_first_invest = self.data["returns"].index[self.data["returns"].index.get_loc(first_invest_date) + retrain_years * 252]

        test_periods.append((first_invest_date, end_date_first_invest))

        n_periods = (self.data["returns"][self.data["returns"].index >= last_date_1st_training].shape[0]) // (retrain_years * 252)

        training_date = first_invest_date

        for i in range(n_periods - 1):
            training_date, end_training_date = training_date, self.data["returns"].index[self.data["returns"].index.get_loc(training_date) + retrain_years * 252]
            invest_date, end_invest_date = end_training_date, self.data["returns"].index[self.data["returns"].index.get_loc(end_training_date) + retrain_years * 252]

            training_periods.append((training_date, end_training_date))
            test_periods.append((invest_date, end_invest_date))

            training_date = invest_date

        return training_periods, test_periods
    
    def _compute_data(self, start, end, training=True, overlap=True, rolling_window=50):
        """
        Calcule les fenêtres glissantes sur les données pour l'entraînement et la validation
        """
        rolling_data = []
        idx_start, idx_end = self.data["returns"].index.get_loc(start), self.data["returns"].index.get_loc(end)

        if training:
            data = self.data["returns"].iloc[idx_start:idx_end, :].copy()

            if overlap:
                for i in range(len(data), rolling_window + 1, -1):
                    rolling_data.append((data.iloc[i - rolling_window - 1: i - 1, :], data.iloc[i - rolling_window: i, :]))

                return rolling_data[::-1]

            else:
                for i in range(len(data), rolling_window + 1, -rolling_window):
                    rolling_data.append((data.iloc[i - rolling_window - 1: i - 1, :], data.iloc[i - rolling_window: i, :]))

                return rolling_data[::-1]

        else:
            for i in range(idx_start, idx_end):
                rolling_data.append((self.data["returns"].iloc[i - rolling_window: i, :]))

            return rolling_data
        
    def load_training_periods(self, initial_train_years=4, retrain_years=2):
        self.periods_train, self.periods_invest = self._generate_training_periods(initial_train_years, retrain_years)

    def loader_period(self, period_index=0, rolling_window=50, batch_size=32, overlap=True, shuffle=True,
                      verbose=False):
        """
        Crée un DataLoader pour une période d'entraînement donnée.
        Cette fonction génère un DataLoader pour une seule période de formation spécifique.
        """

        start_training, end_training = self.periods_train[period_index]
        start_invest, end_invest = self.periods_invest[period_index]

        if verbose:
            print(f'Training period from {start_training} to {end_training}')
            print(f'Investment period from {start_invest} to {end_invest}')

        data_training = self._compute_data(start=start_training, end=end_training, rolling_window=rolling_window, training=True, overlap=overlap)
        data_invest = self._compute_data(start=start_invest, end=end_invest, rolling_window=rolling_window, training=False)

        X_tensor = torch.tensor([df[0].values for df in data_training], dtype=torch.float32)
        Y_tensor = torch.tensor([df[1].values for df in data_training], dtype=torch.float32)
        X_test = torch.tensor([df.values for df in data_invest], dtype=torch.float32)

        dataset = TensorDataset(X_tensor, Y_tensor)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader, X_test, (start_training, end_training, start_invest, end_invest)

    def prices(self):

        return self.data["prices"]
    
    def returns(self):

        return self.data["returns"]

    def return_prices(self):

        return self.data["return_prices"]

