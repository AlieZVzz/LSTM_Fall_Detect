import numpy as np
import torch.nn as nn
import torch
from dataset import state_dataset
from sklearn import svm
from sklearn.metrics import classification_report

# 这些是超参数，我都帮你调好了一般不要动
sequence_length = 256
input_size = 3
hidden_size = 256
num_layers = 3
num_classes = 2
num_epochs = 5
learning_rate = 1e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据集，可以会比较久
print("loading data")
train_dataset = state_dataset('MobiFall_Dataset_v2.0/')
test_dataset = state_dataset('test/')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=256,
                                          shuffle=True)


# 构建一个LSTM模型
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


# 模型实例化
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
weights = [0.005, 0.1]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

# LSTM训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (moves, labels) in enumerate(train_loader):
        # moves = moves.reshape(-1, sequence_length, input_size).to(device)
        moves = moves.to(device)
        labels = labels.to(device)
        # print(moves)

        # Forward pass
        outputs = model(moves)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 保存模型
torch.save(model, 'model.pt')

# 测试
lstm_model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
lstm_model.load_state_dict(torch.load('model.ckpt'), strict=True)
# 加载SVM模型
clf = svm.SVC(gamma='auto')

with torch.no_grad():
    correct = 0
    total = 0
    svm_x = []
    svm_y = []
    for moves, labels in test_loader:
        moves = moves.to(device)
        labels = labels.to(device)
        outputs = lstm_model(moves)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print(outputs)
        # print(classification_report(predicted.to('cpu').tolist(), outputs.to('cpu').tolist()))
        correct += (predicted == labels).sum().item()
        if len(set(labels.to('cpu').tolist())) > 1:
            clf.fit(predicted.to('cpu').reshape(-1, 1), labels.to('cpu'))
        print('Test Accuracy of the model on the batch test state: {} %'.format(100 * correct / total))
