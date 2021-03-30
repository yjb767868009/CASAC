import torch.nn as nn


class ContactPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(#nn.Dropout(0.1),
                                    nn.Linear(1280, 512),
                                    nn.ELU())
        self.layer2 = nn.Sequential(#nn.Dropout(0.1),
                                    nn.Linear(512, 256),
                                    nn.ELU())
        self.layer3 = nn.Sequential(#nn.Dropout(0.1),
                                    nn.Linear(256, 32),
                                    nn.ELU())
        self.layer4 = nn.Sequential(#nn.Dropout(0.1),
                                    nn.Linear(32, 5),
                                    )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
