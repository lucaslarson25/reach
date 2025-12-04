"""
PyTorch model for gesture classification from pose sequences.
Uses LSTM to process temporal sequences of hand/arm keypoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureClassifier(nn.Module):
    """
    LSTM-based classifier for gesture recognition from pose sequences.
    
    Input: (batch_size, sequence_length, num_features)
    - sequence_length: variable length sequences
    - num_features: 144 (hand1: 63, hand2: 63, pose_arms: 18)
    
    Output: (batch_size, num_classes)
    - num_classes: 3 (handshake, fist_bump, high_five)
    """
    
    def __init__(self, input_size=144, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(GestureClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        # Bidirectional LSTM outputs 2 * hidden_size
        self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional tensor of sequence lengths for packed sequences
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Pack sequences if lengths provided (for variable length sequences)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Use the last output from the sequence (or mean pooling)
        # For bidirectional LSTM, concatenate forward and backward hidden states
        if lengths is not None:
            # Get the last valid output for each sequence
            batch_size = lstm_out.size(0)
            last_outputs = []
            for i in range(batch_size):
                last_idx = lengths[i] - 1
                last_outputs.append(lstm_out[i, last_idx, :])
            output = torch.stack(last_outputs)
        else:
            # Use the last timestep
            output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc1(output)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.dropout1(output)
        
        output = self.fc2(output)
        output = self.bn2(output)
        output = F.relu(output)
        output = self.dropout2(output)
        
        logits = self.fc3(output)
        
        return logits


class SimpleGestureClassifier(nn.Module):
    """
    Simpler CNN-based classifier using 1D convolutions for sequence processing.
    Alternative architecture that may work well for fixed-length sequences.
    """
    
    def __init__(self, input_size=144, num_classes=3, dropout=0.3):
        super(SimpleGestureClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Transpose for Conv1d: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove sequence dimension
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.fc2(x)
        
        return logits


def create_model(model_type='lstm', **kwargs):
    """
    Factory function to create a gesture classifier model.
    
    Args:
        model_type: 'lstm' or 'cnn'
        **kwargs: Model hyperparameters
    
    Returns:
        Model instance
    """
    if model_type == 'lstm':
        return GestureClassifier(**kwargs)
    elif model_type == 'cnn':
        return SimpleGestureClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test model
    model = GestureClassifier()
    
    # Test with variable length sequences
    batch_size = 4
    seq_len = 50
    input_size = 144
    
    x = torch.randn(batch_size, seq_len, input_size)
    lengths = torch.tensor([50, 45, 40, 35])
    
    print(f"Input shape: {x.shape}")
    output = model(x, lengths)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

