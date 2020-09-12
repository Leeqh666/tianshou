import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Embedding(nn.Module):
    def __init__(self, c, h, w, device='cpu'):
        super(Embedding, self).__init__()
        self.device = device

        # def conv2d_size_out(size, kernel_size=5, stride=2):
        #     return (size - (kernel_size - 1) - 1) // stride + 1

        # def conv2d_layers_size_out(size,
        #                            kernel_size_1=8, stride_1=4,
        #                            kernel_size_2=4, stride_2=2,
        #                            kernel_size_3=3, stride_3=1):
        #     size = conv2d_size_out(size, kernel_size_1, stride_1)
        #     size = conv2d_size_out(size, kernel_size_2, stride_2)
        #     size = conv2d_size_out(size, kernel_size_3, stride_3)
        #     return size

        # convw = conv2d_layers_size_out(w)
        # convh = conv2d_layers_size_out(h)
        # linear_input_size = convw * convh * 64
        # print(linear_input_size)

        # self.net = nn.Sequential(
        #     nn.Conv2d(c, 32, kernel_size=8, stride=4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(linear_input_size, 512),
        #     nn.Sigmoid()
        # )
        self.net = nn.Sequential(
                        nn.Conv2d(c,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64,64,kernel_size=3, stride=2, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64,64,kernel_size=3,stride=2,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.Sigmoid()
                        )

    def forward(self, x, state=None, info={}):
        r"""x -> Q(x, \*)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        # print(type(x))
        return self.net(x), state

class Prediction(nn.Module):
    def __init__(self, c, h, w, action_shape, device='cpu'):
        super(Prediction, self).__init__()
        self.device = device

        self.embed_net =Embedding(c, h, w, device)

        self.net = nn.Sequential(
            # nn.Conv2d(128, 64, kernel_size=1),
            # nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2048, 512),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, np.prod(action_shape)),
            nn.LogSoftmax(dim=1)
        )

        numel_list = [p.numel() for p in self.embed_net.parameters()]
        print(sum(numel_list), numel_list)
        numel_list = [p.numel() for p in self.net.parameters()]
        print(sum(numel_list), numel_list)

    def forward(self, x1, x2, state=None, info={}):
        r"""x -> Q(x, \*)"""
        if not isinstance(x1, torch.Tensor):
            x = torch.tensor(x1, device=self.device, dtype=torch.float32)
        if not isinstance(x2, torch.Tensor):
            x = torch.tensor(x2, device=self.device, dtype=torch.float32)  
        
        x1 = self.embed_net(x1)
        x2 = self.embed_net(x2)
        # print(type(x1))
        # print(type(x2))
        # print(type(x1[0]), x1[0].size())
        # print(type(x2[0]), x2[0].size())

        x = torch.cat((x1[0], x2[0]), dim=1)
        # print(x.size())
        # print(type(x))     
        return self.net(x), x1[0], x2[0], state
