import torch.nn as nn
import torch


class PitchGraderMLPEmbed(nn.Module):
    def __init__(
        self,
        pitch_features,
        n_batters,
        n_pitchers,
        embed_dim=16,
        hidden_dim=64,
        output_dim=1,
    ):
        super(PitchGraderMLPEmbed, self).__init__()
        self.batter_embedding = nn.Embedding(n_batters, embed_dim)
        self.pitcher_embedding = nn.Embedding(n_pitchers, embed_dim)
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(pitch_features, hidden_dim)
        self.output_layer = nn.Linear(2 * embed_dim + hidden_dim, output_dim)
        self.hidden_layer = nn.Linear(
            2 * embed_dim + hidden_dim, 2 * embed_dim + hidden_dim
        )

    def forward(self, x):
        """
        We take in the pitch features and output an probability of class [ball, strike, in-play]
        Likelihood of in-play is considedered a hittable pitch
        """
        batter = x[:, -2].type(torch.LongTensor).to(x.device)
        pitcher = x[:, -1].type(torch.LongTensor).to(x.device)
        pitch = x[:, :-2]
        batter = self.batter_embedding(batter)
        pitcher = self.pitcher_embedding(pitcher)
        pitch = self.input_layer(pitch)
        pitch = self.activation(pitch)
        combined = torch.cat((batter, pitcher, pitch), dim=1)
        combined = self.hidden_layer(combined)
        combined = self.activation(combined)
        return self.output_layer(combined)
