import numpy as np
import torch
import torch.nn as nn


def get_linear_sequential(input_dims, output_dim, hidden_dims, p_drop=None):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)


def get_convolution_sequential(input_dims, hidden_dims, kernel_dim, p_drop=None):
    channel_dim = input_dims[2]
    dims = [channel_dim] + hidden_dims
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_dim, padding=(kernel_dim - 1) // 2))
        layers.append(nn.ReLU())
        if p_drop is not None:
            layers.append(nn.Dropout(p=p_drop))
        layers.append(nn.MaxPool2d(2, padding=0))
    return nn.Sequential(*layers)


class LeNet5(nn.Module):
    def __init__(self, output_dim=10, input_dim=3):
        super(LeNet5, self).__init__()
        self.ema = None
        input_lin = 400 if input_dim == 3 else 256
        self.conv1 = nn.Conv2d(input_dim, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(input_lin, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, output_dim)
        # self.relu5 = nn.ReLU()
        self.emb_dim = 84

    def forward(self, x, im_u_w=None, im_u_s=None):

        if im_u_w is None and im_u_s is None:
            y = self.conv1(x)
            y = self.relu1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.relu2(y)
            y = self.pool2(y)
            y = y.view(y.shape[0], -1)
            y = self.fc1(y)
            y = self.relu3(y)
            y = self.fc2(y)
            emb = self.relu4(y)
            y = self.fc3(emb)
            # y = self.relu5(y)
            return y, emb

        batch_size_x = x.shape[0]
        inputs = torch.cat((x, im_u_s))

        y = self.conv1(inputs)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        emb = self.relu4(y)
        y = self.fc3(emb)
        logits_x = y[:batch_size_x]
        logits_u_s = y[batch_size_x:]
        with torch.no_grad():  # no gradient to ema model
            y = self.conv1(im_u_w)
            y = self.relu1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.relu2(y)
            y = self.pool2(y)
            y = y.view(y.shape[0], -1)
            y = self.fc1(y)
            y = self.relu3(y)
            y = self.fc2(y)
            emb = self.relu4(y)
            logits_u_w = self.fc3(emb)

        return logits_x, logits_u_w, logits_u_s

    def get_embedding_dim(self):
        return self.emb_dim


class LinearSeq(nn.Module):
    def __init__(self, input_dims, output_dim, linear_hidden_dims, p_drop):
        super().__init__()
        self.input_dims, self.output_dim, self.linear_hidden_dims = input_dims, output_dim, linear_hidden_dims
        self.p_drop = p_drop
        self.linear = get_linear_sequential(
            input_dims=self.input_dims,
            hidden_dims=self.linear_hidden_dims,
            output_dim=self.output_dim,
            p_drop=self.p_drop
        )

    def forward(self, x: torch.FloatTensor):
        batch_size = x.size(0)
        output = self.linear(x.view(batch_size, -1))
        feature_extractor = torch.nn.Sequential(*list(self.linear.children())[:-1])
        features = feature_extractor(x.view(batch_size, -1))
        return output, features


class ConvLinSeq(nn.Module):
    def __init__(self, input_dims, output_dim, linear_hidden_dims, p_drop, kernel_dim,
                 conv_hidden_dims):
        super().__init__()
        self.input_dims, self.output_dim, self.linear_hidden_dims = input_dims, output_dim, linear_hidden_dims
        self.p_drop, self.kernel_dim, self.conv_hidden_dims = p_drop, kernel_dim, conv_hidden_dims
        self.convolutions, self.linear = self.init_model()

    def init_model(self):
        convolutions = get_convolution_sequential(
            input_dims=self.input_dims,
            hidden_dims=self.conv_hidden_dims,
            kernel_dim=self.kernel_dim,
            p_drop=self.p_drop
        )
        lin_input_dims = [self.conv_hidden_dims[-1] * (self.input_dims[0] // 2 ** len(self.conv_hidden_dims)) * (
            self.input_dims[1] // 2 ** len(self.conv_hidden_dims))]
        linear = get_linear_sequential(
            input_dims=lin_input_dims,
            hidden_dims=self.linear_hidden_dims,
            output_dim=self.output_dim,
            p_drop=self.p_drop
        )
        return convolutions, linear

    def forward(self, x: torch.FloatTensor):
        batch_size = x.size(0)
        conv_out = self.convolutions(x)
        output = self.linear(conv_out.view(batch_size, -1))
        feature_extractor = torch.nn.Sequential(*list(self.linear.children())[:-1])
        features = feature_extractor(conv_out.view(batch_size, -1))
        return output, features
