import torch
import math
import torch.nn as nn

class ScaledSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert dim % 2 == 0
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim, dtype=torch.float32) / half_dim 
        inv_freq = theta ** (-freq_seq)
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        seq_len, device = x.shape[1], x.device
        pos = torch.arange(seq_len, device=device)
        emb = torch.einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        pos = pos * self.scale
        if pos.ndim == 1:
            pos = pos.unsqueeze(1)
        return pos
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        """
        
        Args:
            dim_model (int): embedding dimension
            dropout_p (float): dropout probability
            max_len (int): length of sequences 
        """
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding)
    
############# STT #############
class STT(nn.Module):
    def __init__(self, transformer_dim, num_heads, num_layers, seq_len, dropout_p=0.1, 
                 pos='learnable', mlp_dim=1024, activate='gelu', temporal_window=2):
        super().__init__()
        self.pos = pos
        
        # Setup positional encodings
        if pos == 'learnable':
            self.positional_encodings = nn.Parameter(torch.zeros(seq_len, transformer_dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        elif pos == 'sinusoidal':
            self.positional_encodings = ScaledSinusoidalPositionalEncoding(transformer_dim)
        elif pos == 'classic':  # or 'standard'
            self.positional_encodings = PositionalEncoding(transformer_dim, dropout_p, seq_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos}")
        # Create transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attention': FrameAttentionBlock(
                    dim=transformer_dim, 
                    num_heads=num_heads,
                    temporal_window=temporal_window
                ),
                'feed_forward': nn.Sequential(
                    nn.Linear(transformer_dim, mlp_dim),
                    nn.GELU() if activate == 'gelu' else nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(mlp_dim, transformer_dim)
                )
            }) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(transformer_dim)
        
    def forward(self, x):
        if self.pos == 'learnable':
            # For learnable embeddings, add the parameter tensor directly
            x = x + self.positional_encodings.unsqueeze(0)  # Add batch dimension
        elif self.pos == 'sinusoidal':
            # For sinusoidal, call the module
            x = x + self.positional_encodings(x)
        else:
            # For other types, call the module
            x = self.positional_encodings(x)
            
        # Pass through transformer layers
        for layer in self.layers:
            # Apply cross attention
            attended = layer['cross_attention'](x)
            # Apply feed-forward network with residual connection
            x = attended + layer['feed_forward'](attended)
            
        return self.norm(x)

class FrameAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, temporal_window=2):
        super().__init__()
        self.num_heads = num_heads
        self.intra_frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.inter_frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_window = temporal_window
        
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        
        # Intra-frame attention (local attention within each frame)
        x_intra = self.intra_frame_attn(x, x, x)[0]
        x = self.norm1(x + x_intra)
        
        # Inter-frame attention (global attention between frames with temporal masking)
        temporal_mask = self.create_temporal_mask(seq_len, self.temporal_window)
        temporal_mask = temporal_mask.to(x.device)
        
        # Expand mask for multi-head attention:
        # 1. Add batch dimension: (1, seq_len, seq_len)
        # 2. Expand to match number of heads: (batch_size * num_heads, seq_len, seq_len)
        expanded_mask = temporal_mask.unsqueeze(0).expand(batch_size * self.num_heads, -1, -1)
        
        # Apply inter-frame attention with temporal masking
        x_inter = self.inter_frame_attn(x, x, x, attn_mask=expanded_mask)[0]
        x = self.norm2(x + x_inter)
        
        return x
    
    def create_temporal_mask(self, seq_len, window):
        """
        Create a temporal mask that allows attention only within a local window.
        
        Args:
            seq_len (int): Length of the sequence
            window (int): Size of the temporal window (one-sided)
            
        Returns:
            torch.Tensor: Mask tensor of shape (seq_len, seq_len) where 0 indicates
                         allowed attention and -inf blocks attention
        """
        # Start with all -inf (block all attention)
        mask = torch.full((seq_len, seq_len), float('-inf'))
        
        # For each position, allow attention to nearby frames within the window
        for i in range(seq_len):
            # Calculate valid range for attention
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            mask[i, start:end] = 0.0  # Allow attention within window
            
        return mask
################################################

############# Vanilla Transformer #############
def exists(val):
    return val is not None

class Transformer(nn.Module):
    def __init__(self, transformer_dim=128, num_heads=8, num_layers=6, seq_len=13, dropout_p=0.1, pos='sinusoidal', mlp_dim=1024, activate='gelu'):
        """
        Args:
            transformer_dim (int, optional): embedding dimension of transformer layers . Defaults to 128.
            num_heads (int, optional): num transformer heads. Defaults to 8.
            num_layers (int, optional): num of encoder layers. Defaults to 6.
            seq_len (int, optional): length of sequences. Defaults to 5.
            dropout_p (float, optional): dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.pos = pos
        if pos == 'learnable':
            self.positional_encodings = nn.Parameter(torch.zeros(seq_len, transformer_dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        elif pos == 'sinusoidal':
            self.positional_encodings = ScaledSinusoidalPositionalEncoding(transformer_dim)
        else:
            self.positional_encodings = PositionalEncoding(transformer_dim, dropout_p, seq_len)
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads,  batch_first=True, dropout=dropout_p, dim_feedforward=mlp_dim, activation=activate)
        layer_norm = nn.LayerNorm(transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=layer_norm)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
             
    def forward(self, x):
        """Expect input x to be of shape (batch_size, sequence_length, transformer_dim)"""
        if self.pos in ['learnable', 'sinusoidal']:
            x = x + self.positional_encodings.unsqueeze(0)  # Add batch dimension
        else:
            x = self.positional_encodings(x)
        x = self.transformer_encoder(x)
        return x
################################################