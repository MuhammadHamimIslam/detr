import torch
from torch import nn

class MLP(nn.Module):
   """ MLP layer for the box regression """
   def __init__(self, input_dim=256, hidden_dim=256, output_dim=4, num_layers=3):
       super().__init__()

       layers = []

       for i in range(num_layers - 1):
           layers.append(nn.Linear(
               input_dim if i == 0 else hidden_dim,
               hidden_dim
             )
           )
           layers.append(nn.ReLU())
           layers.append(nn.Droput(0.1))

       layers.append(nn.Linear(hidden_dim, output_dim))

       self.layers = nn.Sequential(*layers)


   def forward(self, x):
       return self.layers(x)

class DETRHead(nn.Module):
   """ Full head (classification and Box) for detr """
   def __init__(self, num_classes, input_dim, hidden_dim, num_layers=3):
       super().__init__()

       self.cls = nn.Linear(input_dim, num_classes + 1)
       self.box_head = MLP(input_dim, hidden_dim, 4, num_layers)

   def forward(self, x):
       boxes = self.box_head(x)
       logits = self.cls(x)
       boxes = boxes.sigmoid()

       return logits, boxes