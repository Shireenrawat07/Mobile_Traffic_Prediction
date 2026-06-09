import torch
import torch.nn as nn

class CnnTrafficPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_channels=64, output_size=1):
        super(CnnTrafficPredictor, self).__init__()
        
        # First 1D Convolutional Layer
        # in_channels maps to your input_size (features per time step)
        self.conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=hidden_channels, 
            kernel_size=3, 
            padding=1 # Padding=1 keeps the sequence length the same for kernel=3
        )
        self.relu = nn.ReLU()
        
        # Second 1D Convolutional Layer
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels, 
            out_channels=hidden_channels * 2, 
            kernel_size=3, 
            padding=1
        )
        
        # Adaptive Average Pooling
        # This reduces the entire sequence length dimension down to 1 value per channel,
        # making the model robust to varying sequence lengths (just like taking the last LSTM output).
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Layer for the final prediction
        self.fc = nn.Linear(hidden_channels * 2, output_size)

    def forward(self, x):
        # 1. Reshape input from (Batch, Seq_Len, Features) to (Batch, Features, Seq_Len)
        x = x.transpose(1, 2)
        
        # 2. Pass through Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        # 3. Global Pooling: Shape becomes (Batch, Channels, 1)
        x = self.global_pool(x)
        
        # 4. Flatten the output for the Linear layer: Shape becomes (Batch, Channels)
        x = x.squeeze(-1)
        
        # 5. Final Prediction
        out = self.fc(x)
        return out

def load_cnn_model_from_checkpoint(checkpoint_path, input_size=1, output_size=1, device='cpu'):
    """Utility function to load the CNN model, similar to your LSTM loader."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Simple hidden channel detection based on the first convolution layer's weights
    conv1_weight_shape = checkpoint['conv1.weight'].shape
    hidden_channels = conv1_weight_shape[0]
    
    print(f"Detected hidden_channels={hidden_channels} from checkpoint.")
    
    model = CnnTrafficPredictor(
        input_size=input_size, 
        hidden_channels=hidden_channels, 
        output_size=output_size
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model