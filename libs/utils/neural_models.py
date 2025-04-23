import torch
from sklearn.utils import shuffle
from torch import nn

# find possible cuda cores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    def __init__(self, x, y, batch_size=128, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_idx = 0
        self.data_size = x.shape[0]
        if self.shuffle:
            self.reset()

    def reset(self):
        self.x, self.y = shuffle(self.x, self.y)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_idx >= self.data_size:
            if self.shuffle:
                self.reset()
            self.start_idx = 0
            raise StopIteration

        batch_x = self.x[self.start_idx : self.start_idx + self.batch_size, :]
        batch_y = self.y[self.start_idx : self.start_idx + self.batch_size, :]

        batch_x = torch.tensor(batch_x, dtype=torch.float, device=device)
        batch_y = torch.tensor(batch_y, dtype=torch.float, device=device)

        self.start_idx += self.batch_size

        return (batch_x, batch_y)


# defining MLP model
# generally out_dim is more than 1, but this model only allows 1.
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.drop_out(x)
        x = self.linear2(x)
        # x = x.squeeze(1)
        return x
