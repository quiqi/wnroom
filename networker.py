import torch
import torch.nn as nn
import torch.nn.functional as F


class DANN(nn.Module):
    def __init__(self, n_domains: int = 11, n_classes: int = 2):
        """
        DANN 神经网络
        :param n_domains: 领域个数
        :param n_classes: 类别个数
        """
        super(DANN, self).__init__()
        self.n_domains = n_domains
        self.n_classes = n_classes

        # Feature extractor 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(2500, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Domain classifier 域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_domains)
        )

        # Label predictor  标签分类器
        self.label_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x, alpha):
        # Feature extraction
        features = self.feature_extractor(x)

        # Gradient reversal layer (GRL)
        grl_features = GradientReversalLayer.apply(features, alpha)

        # Domain classification
        domain_logits = self.domain_classifier(grl_features)

        # Label prediction
        label_logits = self.label_predictor(features)

        return domain_logits, label_logits


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers):
        super(FullyConnectedNetwork, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()

        # Hidden layers
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input layer
        x = nn.functional.relu(self.input_layer(x))

        # Hidden layers
        for hidden_layer in self.hidden_layers:
            x = nn.functional.relu(hidden_layer(x))

        # Output layer
        x = self.output_layer(x)
        return x


class FC(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        # Feature extractor 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(2500, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Label predictor  标签分类器
        self.label_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.label_predictor(x)
        return x
