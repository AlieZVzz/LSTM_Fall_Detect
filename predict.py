import numpy as np
import torch.nn as nn
import sys
from sklearn.preprocessing import StandardScaler
import torch

sequence_length = 256
input_size = 3
hidden_size = 256
num_layers = 3
num_classes = 2
num_epochs = 5
learning_rate = 1e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# 数据预处理
scaler = StandardScaler()

lstm_model = torch.load('model.pt', map_location=torch.device('cpu'))

test_path = sys.argv[1]



def get_sensor_data(file_path, frequency=8):
    acc_file = open(file_path, 'r', encoding='utf-8')
    # gyro_file = open(file_path.replace('acc','gyro'), 'r', encoding='utf-8')
    # ori = open(file_path.replace('acc','ori'), 'r', encoding='utf-8')
    data = []
    for idx, i in enumerate(acc_file.readlines()):
        if idx % frequency == 0:
            acc_data = i.strip().split(', ')
            acc_data = [np.float(i) for i in acc_data]
            # print(acc_data)
            data.append(acc_data)
            # print(data)

    return scaler.fit_transform(data)


moves = torch.tensor(get_sensor_data(test_path), dtype=torch.float).unsqueeze(dim=0)

with torch.no_grad():
    moves = moves.to(device)
    outputs = lstm_model(moves)
    _, predicted = torch.max(outputs.data, 1)
    print(torch.nn.functional.softmax(outputs).squeeze().to('cpu').tolist())

