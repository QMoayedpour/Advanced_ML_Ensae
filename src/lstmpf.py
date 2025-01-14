import torch
import torch.nn as nn


class NN_Sharpe(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=4, num_layers=1, model_name='LSTM',
                 temperature=0.1):
        super(NN_Sharpe, self).__init__()

        self.num_layers = num_layers
        self.model_name = model_name

        if self.model_name == 'LSTM':
            self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif self.model_name == 'RNN':
            self.model = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif self.model_name == "GRU":
            self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("model_name not valid, select between ['LSTM', 'RNN', 'GRU']") 
        self.linear = nn.Linear(hidden_size, output_size)
        self.temperature = temperature

    def get_alloc(self, x):

        # x : shape[batch_size, seq_length, input_size]

        # output : [batch_size, sequ_length, hidden_size]
        # cn : [num_layers, batch_size, output_size], it's the final state of each layer if proj_size>0 (only for LSTM)
        # hn : [num_layers, batch_size, output_size], it's the final state of each layer

        if self.model_name == "LSTM":
            # if num_layers >1, we only keep the last layer for computing loss

            output, (hn,cn) = self.model(x)
            output, (hn,cn) = output[:,:,:], (hn[:, :, :], cn[:, : , :])

        elif self.model_name in ['GRU', 'RNN']:

            output, hn = self.model(x)
            output, hn = output[:,:,:], hn[:, :, :]

        #tanh_output = torch.tanh(output)

        unnormalized_weights = self.linear(output)
        scaled_weights = unnormalized_weights / self.temperature

        normalized_weights = torch.softmax(scaled_weights, dim=-1)
        return normalized_weights

    def get_alloc_last(self, x):

        x = self.get_alloc(x)
        return x[:, -1, :]

    def sharpe_loss(self, weights, y):

        # weights : [batch_size, seq_length, output_size]
        # y : [batch_size, seq_length, output_size]

        # returns : [batch_size, seq_length]
        returns = torch.sum(weights*y, dim=-1)

        # mean_returns/std_returns : [batch_size]
        mean_returns = returns.mean(dim=-1)
        std_returns = returns.std(dim=-1) + 1e-12  # No division by 0

        mean_sharpe_ratio = mean_returns / std_returns  # shape [batch_size]

        return -mean_sharpe_ratio  # maximizing the sharpe ratio
    
    def forward(self, x, y):

        x = self.get_alloc(x)
        loss = self.sharpe_loss(x, y).mean()

        return loss

