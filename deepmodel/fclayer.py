import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ThreeLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size_1, hidden_size_2, num_classes):

        super(ThreeLayerNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.input_dim, self.hidden_size_1)
        # self.dropout1 = nn.Dropout(p=0.3)
        # self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        # self.dropout2 = nn.Dropout(p=0.3)
        # self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size_2, self.num_classes)
        # self.dropout3 = nn.Dropout(p=0.25)
        # self.relu3 = nn.ReLU()

    def forward(self, x):
        out = None
        # x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = self.dropout3(x)
        # x = F.relu(x)
        out = x

        return out


class FourLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size_1, hidden_size_2, hidden_size_3, num_classes):

        super(FourLayerNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.input_dim, self.hidden_size_1)
        # self.dropout1 = nn.Dropout(p=0.3)
        # self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        # self.dropout2 = nn.Dropout(p=0.3)
        # self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        # self.dropout3 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(self.hidden_size_3, self.num_classes)

        # self.relu3 = nn.ReLU()

    def forward(self, x):
        out = None
        # x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = self.dropout3(x)
        x = F.relu(x)
        x = self.fc4(x)
        out = x

        return out

