"""
Training script for gesture classifier.
Loads collected gesture data and trains a PyTorch model.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse
from tqdm import tqdm

# Import model
import sys
# Add parent directory to path to import models
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from models.gesture_classifier import create_model


class GestureDataset(Dataset):
    """Dataset class for gesture sequences."""
    
    def __init__(self, sequences, labels, scaler=None, max_length=None):
        self.sequences = sequences
        self.labels = labels
        self.scaler = scaler
        self.max_length = max_length
        
        # Normalize sequences if scaler provided
        if scaler is not None:
            self.sequences = self._normalize_sequences(sequences)
        
        # Pad sequences to max_length if specified
        if max_length is not None:
            self.sequences = self._pad_sequences(sequences, max_length)
    
    def _normalize_sequences(self, sequences):
        """Normalize sequences using the provided scaler."""
        normalized = []
        for seq in sequences:
            # Reshape for scaler: (seq_len, features) -> (seq_len * features,)
            # Then reshape back
            seq_flat = seq.reshape(-1, seq.shape[-1])
            seq_normalized = self.scaler.transform(seq_flat)
            normalized.append(seq_normalized)
        return normalized
    
    def _pad_sequences(self, sequences, max_length):
        """Pad sequences to max_length."""
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                # Pad with zeros
                pad_length = max_length - len(seq)
                padding = np.zeros((pad_length, seq.shape[-1]))
                padded_seq = np.vstack([seq, padding])
            else:
                # Truncate if too long
                padded_seq = seq[:max_length]
            padded.append(padded_seq)
        return padded
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        length = torch.LongTensor([len(self.sequences[idx])])[0]
        
        return sequence, label, length


def load_gesture_data(data_dir='scenes/image_recognition/data'):
    """Load all collected gesture data."""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    sequences = []
    labels = []
    gesture_to_idx = {'handshake': 0, 'fist_bump': 1, 'high_five': 2}
    
    # Load data from each gesture class
    for gesture_name, gesture_idx in gesture_to_idx.items():
        gesture_dir = data_dir / gesture_name
        
        if not gesture_dir.exists():
            print(f"Warning: No data found for {gesture_name}")
            continue
        
        json_files = list(gesture_dir.glob('*.json'))
        print(f"Loading {len(json_files)} sequences for {gesture_name}...")
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                sequence = np.array(data['sequence'], dtype=np.float32)
                sequences.append(sequence)
                labels.append(gesture_idx)
    
    if len(sequences) == 0:
        raise ValueError("No gesture data found! Please collect data first.")
    
    print(f"\nLoaded {len(sequences)} total sequences")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return sequences, labels, gesture_to_idx


def train_epoch(model, dataloader, criterion, optimizer, device, use_lengths=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels, lengths in tqdm(dataloader, desc="Training"):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        if use_lengths and hasattr(model, 'lstm'):
            outputs = model(sequences, lengths)
        else:
            outputs = model(sequences)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, use_lengths=True):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels, lengths in tqdm(dataloader, desc="Validating"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            if use_lengths and hasattr(model, 'lstm'):
                outputs = model(sequences, lengths)
            else:
                outputs = model(sequences)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train gesture classifier')
    parser.add_argument('--data-dir', type=str, default='scenes/image_recognition/data',
                        help='Directory containing gesture data')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'cnn'],
                        help='Model architecture type')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--max-length', type=int, default=None,
                        help='Maximum sequence length (None for variable length)')
    parser.add_argument('--output-dir', type=str, default='scenes/image_recognition/models',
                        help='Directory to save trained model')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading gesture data...")
    sequences, labels, gesture_to_idx = load_gesture_data(args.data_dir)
    
    # Prepare data
    # Flatten all sequences for normalization
    all_features = np.vstack([seq for seq in sequences])
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Determine max length if not specified
    if args.max_length is None:
        max_length = max(len(seq) for seq in sequences)
        print(f"Using variable length sequences (max: {max_length})")
    else:
        max_length = args.max_length
        print(f"Using fixed length sequences: {max_length}")
    
    # Split data: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 of remaining = 20% of total
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = GestureDataset(X_train, y_train, scaler=scaler, max_length=max_length)
    val_dataset = GestureDataset(X_val, y_val, scaler=scaler, max_length=max_length)
    test_dataset = GestureDataset(X_test, y_test, scaler=scaler, max_length=max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    model_kwargs = {
        'input_size': 144,
        'num_classes': 3,
    }
    
    if args.model_type == 'lstm':
        model_kwargs.update({
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
        })
    
    model = create_model(args.model_type, **model_kwargs)
    model = model.to(device)
    
    print(f"\nModel: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_lengths=(args.max_length is None)
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            use_lengths=(args.max_length is None)
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            print(f"New best validation accuracy: {val_acc:.2f}%")
    
    # Load best model and evaluate on test set
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print("Evaluating on test set...")
    
    model.load_state_dict(best_model_state)
    test_loss, test_acc = validate(
        model, test_loader, criterion, device,
        use_lengths=(args.max_length is None)
    )
    
    print(f"\n{'='*60}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*60}")
    
    # Save final model
    model_path = output_dir / 'gesture_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'model_kwargs': model_kwargs,
        'scaler': scaler,
        'gesture_to_idx': gesture_to_idx,
        'max_length': max_length,
        'epoch': best_epoch,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Check if accuracy requirement is met
    if test_acc >= 85.0:
        print(f"✓ Test accuracy ({test_acc:.2f}%) meets requirement (≥85%)")
    else:
        print(f"⚠ Test accuracy ({test_acc:.2f}%) is below requirement (≥85%)")
        print("  Consider collecting more training data or adjusting hyperparameters")


if __name__ == '__main__':
    main()

