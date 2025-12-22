import torch
import torch.nn as nn

class TrafficPredictorRNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=3,
        output_size=1,
        bidirectional=True
    ):
        super(TrafficPredictorRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # 🔑 IMPORTANT FIX
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc1 = nn.Linear(rnn_output_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        out, _ = self.rnn(x)

        # Take last time step
        out = out[:, -1, :]   # (batch, hidden_size*2 if bidirectional)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
