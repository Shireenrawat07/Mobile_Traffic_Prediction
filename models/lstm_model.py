import torch
import torch.nn as nn

class TrafficPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(TrafficPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def load_model_from_checkpoint(checkpoint_path, input_size=1, output_size=1, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Filter only weight_ih keys of LSTM
    lstm_keys = [k for k in checkpoint.keys() if k.startswith('lstm.weight_ih_l')]
    if not lstm_keys:
        raise ValueError("No LSTM weight_ih keys found in checkpoint.")
    
    # Determine hidden size
    first_weight = checkpoint[lstm_keys[0]]
    hidden_size = first_weight.shape[0] // 4  # LSTM has 4*hidden_size rows
    
    # Determine number of layers
    num_layers = max([int(k[len('lstm.weight_ih_l'):]) for k in lstm_keys]) + 1

    print(f"Detected hidden_size={hidden_size}, num_layers={num_layers} from checkpoint.")

    # Build model
    model = TrafficPredictor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
