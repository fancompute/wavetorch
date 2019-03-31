import torch
import torch.nn as nn
from torch.nn import functional as F

class CustomRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_first=True, W_scale=1e-1, f_hidden=None):
        super(CustomRNN, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.f_hidden = f_hidden

        self.W1 = nn.Parameter((torch.rand(hidden_size, input_size)-0.5)*W_scale)
        self.W2 = nn.Parameter((torch.rand(hidden_size, hidden_size)-0.5)*W_scale)
        self.W3 = nn.Parameter((torch.rand(output_size, hidden_size)-0.5)*W_scale)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        h1 = torch.zeros(x.shape[0], self.hidden_size)
        ys = []

        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            h1 = (torch.matmul(self.W2, h1.t()) + torch.matmul(self.W1, xi.t())).t() + self.b_h
            if self.f_hidden is not None:
                h1 = getattr(F, self.f_hidden)(h1)
            y = torch.matmul(self.W3, h1.t()).t()
            ys.append(y)

        ys = torch.stack(ys, dim=1)
        return ys

class CustomRes(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_first=True, W_scale=1e-1, f_hidden=None):
        super(CustomRes, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.f_hidden = f_hidden

        self.W1 = torch.nn.Parameter((torch.rand(hidden_size, input_size)-0.5)*W_scale)
        self.W2 = torch.nn.Parameter((torch.rand(hidden_size, hidden_size)-0.5)*W_scale)
        self.W3 = torch.nn.Parameter((torch.rand(output_size, hidden_size)-0.5)*W_scale)
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        h1 = torch.zeros(x.shape[0], self.hidden_size)
        ys = []

        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            hprev = h1
            h1 = (torch.matmul(self.W2, h1.t()) + torch.matmul(self.W1, xi.t())).t() + self.b_h
            if self.f_hidden is not None:
                h1 = getattr(F, self.f_hidden)(h1)
            y = torch.matmul(self.W3, h1.t()).t()
            ys.append(y)
            h1 = h1 + hprev

        ys = torch.stack(ys, dim=1)
        return ys

class CustomLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_first=True, W_scale=1e-1):
        super(CustomLSTM, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        self.W3 = torch.nn.Parameter((torch.rand(output_size, hidden_size)-0.5))

    def forward(self, x):

        # out should have size [N_batch, T, N_hidden] 
        out, hidden = self.lstm(x.unsqueeze(2))
        # print(torch.max(x, 1))
        # print(x[:, 100])
        # print(out[:, 100, 0].detach())
        # ys should have size [N_batch, T, N_classes]
        ys = torch.matmul(out, self.W3.t())

        return ys