from torch import nn
import torch.nn.functional as F 

class LeNetEncoder(nn.Module):
    def __init__(self):
        super( LeNetEncoder, self ).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            
            nn.Conv2d( in_channels = 3, out_channels = 20, kernel_size = 5 ),
            nn.MaxPool2d( kernel_size = 2 ),
            nn.ReLU(),

            nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 5) ,
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size = 2 ),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(50*4*4,500)

    def forward(self,inputs):
        conv_out = self.encoder( inputs)
        feat = self.fc1(input = conv_out.view( -1, 50*4*4 ))
        return feat


class LeNetClassifier(nn.Module):
    def __init__(self):
        super( LeNetClassifier, self ).__init__()
        self.fc2 = nn.Linear( in_features = 500, out_features = 10 )

    def forward(self, inputs):
        out = F.dropout( F.relu(inputs), training= self.training )
        out = self.fc2( out )
        return out
