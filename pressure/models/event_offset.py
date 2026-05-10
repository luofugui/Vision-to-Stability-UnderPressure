import torch
import torch.nn as nn


class EventOffsetRegressor(nn.Module):
    """Lightweight pose-window model for event offset regression."""

    def __init__(
        self,
        num_joints,
        joint_dim,
        num_channels,
        window_frames,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.window_frames = window_frames

        input_dim = num_joints * joint_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.event_embedding = nn.Embedding(2, hidden_dim)
        self.channel_embedding = nn.Embedding(num_channels, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, joint_window, event_type, channel):
        batch, frames, joints, dims = joint_window.shape
        x = joint_window.reshape(batch, frames, joints * dims)
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = x + self.event_embedding(event_type).unsqueeze(1)
        x = x + self.channel_embedding(channel).unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.head(x).squeeze(-1)
