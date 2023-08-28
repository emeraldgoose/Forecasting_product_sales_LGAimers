import torch.nn as nn
from cfg import *

class Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.actv = nn.GELU()
        self.fc = nn.Linear(hidden_size // 2, CFG.predict_size)
    
    def forward(self, inputs):
        out, _ = self.lstm(inputs)
        out, _ = self.lstm2(self.dropout(self.actv(out)))
        out = self.fc(out[:, -1, :])
        return out