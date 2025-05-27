import torch.nn as nn


class PitchGraderMLP(nn.Module):
    def __init__(
        self, pitch_features, n_batters, n_pitchers, hidden_dim=64, output_dim=1
    ):
        super(PitchGraderMLP, self).__init__()
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(pitch_features, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        We take in the pitch features and output an probability of class [ball, strike, in-play]
        Likelihood of in-play is considedered a hittable pitch
        """
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        return self.output_layer(x)
