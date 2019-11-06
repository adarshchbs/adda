from torch import nn
import torch.nn.functional as F 

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim ):
        super( Discriminator, self ).__init__()

        self.restore = False

        self.layer = nn.Sequential(
            nn.Linear( in_features = input_dim, out_features = hidden_dim ),
            nn.ReLU(),
            # nn.Linear(in_features = hidden_dim, out_features = hidden_dim ),
            # nn.ReLU(),
            nn.Linear(in_features = hidden_dim, out_features = output_dim ),
            # nn.LogSoftmax()
        )

    def forward(self, inputs):
        out = self.layer(inputs)
        return out
