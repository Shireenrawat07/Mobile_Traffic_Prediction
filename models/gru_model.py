import torch
import torch.nn as nn

class TrafficPredictorGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(TrafficPredictorGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out


def load_gru_model_from_checkpoint(checkpoint_path, input_size=1, output_size=1, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Filter only weight_ih keys of GRU
    gru_keys = [k for k in checkpoint.keys() if k.startswith('gru.weight_ih_l')]
    if not gru_keys:
        raise ValueError("No GRU weight_ih keys found in checkpoint.")
    
    # Determine hidden size
    first_weight = checkpoint[gru_keys[0]]
    hidden_size = first_weight.shape[0] // 3  # GRU has 3*hidden_size rows
    
    # Determine number of layers
    num_layers = max([int(k[len('gru.weight_ih_l'):]) for k in gru_keys]) + 1

    print(f"Detected hidden_size={hidden_size}, num_layers={num_layers} from checkpoint.")

    # Build model
    model = TrafficPredictorGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
