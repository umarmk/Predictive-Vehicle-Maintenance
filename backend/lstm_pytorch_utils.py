# lstm_pytorch_utils.py
# LSTM time-series utilities using PyTorch 2.6.0+
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Updated train_lstm signature to accept hidden_size and num_layers

def train_lstm(series, seq_length=10, epochs=20, lr=0.001, batch_size=16, hidden_size=32, num_layers=1):
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))
    X, y = create_sequences(series_scaled, seq_length)
    X_train, X_test = X[:-20], X[-20:]
    y_train, y_test = y[:-20], y[-20:]
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    model = LSTMNet(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8)))  # avoid div by zero
    return model, scaler, rmse, mape, y_test, y_pred

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

