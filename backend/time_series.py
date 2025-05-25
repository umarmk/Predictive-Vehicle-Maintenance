# time_series.py
# Advanced time-series forecasting for Predictive Vehicle Maintenance
# Implements multi-feature LSTM, multi-step forecasting, and component-specific degradation prediction

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
import json
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultivariateLSTM(nn.Module):
    """
    Multivariate LSTM model for time-series forecasting with multiple features.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM layer
            num_layers: Number of LSTM layers
            output_size: Number of output features to predict
            dropout: Dropout rate
        """
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TimeSeriesForecaster:
    """
    Advanced time-series forecasting class for vehicle maintenance prediction.
    """

    def __init__(self, models_dir=None):
        """
        Initialize the time-series forecaster.

        Args:
            models_dir: Directory to save trained models
        """
        if models_dir is None:
            self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        else:
            self.models_dir = models_dir

        os.makedirs(self.models_dir, exist_ok=True)
        self.scalers = {}
        self.models = {}
        self.feature_names = []

    def create_sequences(self, data: np.ndarray, seq_length: int, target_cols: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and target values for time series forecasting.

        Args:
            data: Time series data (should be normalized/scaled)
            seq_length: Number of time steps to use for each input sequence
            target_cols: Indices of target columns to predict (default: all columns)

        Returns:
            X: Input sequences of shape (n_samples, seq_length, n_features)
            y: Target values of shape (n_samples, n_target_features)
        """
        n_samples, n_features = data.shape
        X, y = [], []

        if target_cols is None:
            target_cols = list(range(n_features))

        for i in range(n_samples - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, target_cols])

        return np.array(X), np.array(y)

    def prepare_data(self, df: pd.DataFrame, target_cols: List[str], seq_length: int = 10,
                    test_size: float = 0.2, scaling_method: str = 'minmax') -> Dict:
        """
        Prepare data for time-series forecasting.

        Args:
            df: Input dataframe
            target_cols: List of target column names to predict
            seq_length: Sequence length for LSTM input
            test_size: Proportion of data to use for testing
            scaling_method: Method for data scaling ('minmax' or 'standard')

        Returns:
            Dictionary containing prepared data
        """
        # Validate target columns
        for col in target_cols:
            if col not in df.columns:
                raise ValueError(f"Target column {col} not found in dataframe")

        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        self.feature_names = numeric_df.columns.tolist()

        # Get indices of target columns
        target_indices = [self.feature_names.index(col) for col in target_cols]

        # Scale data
        if scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaling method: {scaling_method}. Using MinMaxScaler.")
            scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(numeric_df)
        self.scalers['data_scaler'] = scaler

        # Create sequences
        X, y = self.create_sequences(scaled_data, seq_length, target_indices)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_tensor': X_train_tensor,
            'y_train_tensor': y_train_tensor,
            'X_test_tensor': X_test_tensor,
            'y_test_tensor': y_test_tensor,
            'target_indices': target_indices,
            'feature_names': self.feature_names,
            'target_cols': target_cols
        }

    def train_lstm(self, prepared_data: Dict, model_name: str = 'multivariate_lstm',
                  hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2,
                  lr: float = 0.001, batch_size: int = 32, epochs: int = 50,
                  patience: int = 10) -> Dict:
        """
        Train an LSTM model for time-series forecasting.

        Args:
            prepared_data: Dictionary of prepared data from prepare_data method
            model_name: Name to save the model
            hidden_size: Number of hidden units in LSTM layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            patience: Number of epochs to wait for improvement before early stopping

        Returns:
            Dictionary of training results
        """
        # Extract data
        X_train_tensor = prepared_data['X_train_tensor']
        y_train_tensor = prepared_data['y_train_tensor']
        X_test_tensor = prepared_data['X_test_tensor']
        y_test_tensor = prepared_data['y_test_tensor']

        # Get dimensions
        _, seq_length, n_features = X_train_tensor.shape
        n_outputs = y_train_tensor.shape[1]

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = MultivariateLSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=n_outputs,
            dropout=dropout
        )

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        train_losses = []
        val_losses = []

        logger.info(f"Starting LSTM training for {model_name} with {epochs} epochs")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
                val_losses.append(val_loss)

            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()
            y_true = y_test_tensor.numpy()

        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        logger.info(f"Final metrics: {metrics}")

        # Save model
        self.models[model_name] = model

        # Save model state dict
        model_path = os.path.join(self.models_dir, f'{model_name}_state_dict.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model state dict to {model_path}")

        # Save model architecture info
        model_info = {
            'input_size': n_features,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': n_outputs,
            'dropout': dropout,
            'feature_names': prepared_data['feature_names'],
            'target_cols': prepared_data['target_cols'],
            'target_indices': prepared_data['target_indices'].tolist() if isinstance(prepared_data['target_indices'], np.ndarray) else prepared_data['target_indices'],
            'seq_length': seq_length
        }

        model_info_path = os.path.join(self.models_dir, f'{model_name}_info.json')
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f)
        logger.info(f"Saved model info to {model_info_path}")

        # Save scaler
        scaler_path = os.path.join(self.models_dir, f'{model_name}_scaler.pkl')
        joblib.dump(self.scalers['data_scaler'], scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training History for {model_name}')
        plt.legend()
        plt.savefig(os.path.join(self.models_dir, f'{model_name}_training_history.png'))

        return {
            'model': model,
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'y_pred': y_pred,
            'y_true': y_true
        }

    def predict_future(self, model_name: str, input_sequence: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """
        Make multi-step predictions using a trained LSTM model.

        Args:
            model_name: Name of the trained model
            input_sequence: Input sequence of shape (seq_length, n_features)
            steps_ahead: Number of steps to predict ahead

        Returns:
            Predictions of shape (steps_ahead, n_outputs)
        """
        # Load model info
        model_info_path = os.path.join(self.models_dir, f'{model_name}_info.json')
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)

        # Load model
        model = MultivariateLSTM(
            input_size=model_info['input_size'],
            hidden_size=model_info['hidden_size'],
            num_layers=model_info['num_layers'],
            output_size=model_info['output_size'],
            dropout=model_info['dropout']
        )

        model_path = os.path.join(self.models_dir, f'{model_name}_state_dict.pt')
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Load scaler
        scaler_path = os.path.join(self.models_dir, f'{model_name}_scaler.pkl')
        scaler = joblib.load(scaler_path)

        # Prepare input sequence
        seq_length = model_info['seq_length']
        if input_sequence.shape[0] < seq_length:
            raise ValueError(f"Input sequence length {input_sequence.shape[0]} is less than required {seq_length}")

        # Scale input sequence
        if input_sequence.shape[1] != model_info['input_size']:
            raise ValueError(f"Input sequence has {input_sequence.shape[1]} features, but model expects {model_info['input_size']}")

        scaled_sequence = scaler.transform(input_sequence)

        # Make predictions
        predictions = []
        current_sequence = scaled_sequence[-seq_length:].copy()

        for _ in range(steps_ahead):
            # Prepare input tensor
            input_tensor = torch.tensor(current_sequence.reshape(1, seq_length, -1), dtype=torch.float32)

            # Make prediction
            with torch.no_grad():
                pred = model(input_tensor).numpy()[0]

            # Create a full feature vector for the next time step
            next_step = np.zeros(model_info['input_size'])
            for i, idx in enumerate(model_info['target_indices']):
                next_step[idx] = pred[i]

            # For non-predicted features, use the last values
            for i in range(model_info['input_size']):
                if i not in model_info['target_indices']:
                    next_step[i] = current_sequence[-1, i]

            # Update sequence for next prediction
            current_sequence = np.vstack([current_sequence[1:], next_step])

            # Store prediction
            predictions.append(pred)

        # Convert predictions to numpy array
        predictions = np.array(predictions)

        # Inverse transform predictions (only for target columns)
        inverse_predictions = np.zeros((predictions.shape[0], len(model_info['target_cols'])))
        for i in range(predictions.shape[0]):
            # Create a dummy full-feature array
            dummy = np.zeros((1, model_info['input_size']))
            for j, idx in enumerate(model_info['target_indices']):
                dummy[0, idx] = predictions[i, j]

            # Inverse transform
            inverse_dummy = scaler.inverse_transform(dummy)

            # Extract target values
            for j, idx in enumerate(model_info['target_indices']):
                inverse_predictions[i, j] = inverse_dummy[0, idx]

        return inverse_predictions
