import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn

from pressure.models.embeddings import PoseEmbedder
from pressure.models.transformers import *

############# Util#############
def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def exists(val):
    return val is not None

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.net(x)
    
class TemporalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=n_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        y, _ = self.gru(x)
        y = self.out_proj(y)
        return y

############# Contact Conditioned Pressure Head #############
class ContactConditionedPressureHead(nn.Module):
    """
    Pressure prediction head that is conditioned on contact predictions through cross-attention.
    """
    def __init__(self, input_dim, hidden_dim, pressure_dim, contact_dim, pred_distribution=False):
        super().__init__()
        
        self.contact_projection = nn.Linear(contact_dim, hidden_dim)
        self.pressure_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Residual FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.pressure_gate = nn.Sequential(
            nn.Linear(hidden_dim, pressure_dim),
            nn.Sigmoid()
        )
        
        # Final pressure prediction
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, pressure_dim),
        )

    def forward(self, x, contact_pred):
        # Project inputs
        pressure_query = self.pressure_projection(x).unsqueeze(1)
        contact_key = self.contact_projection(contact_pred).unsqueeze(1)
        
        # Cross attention between pressure and contact features
        attended, _ = self.cross_attention(
            pressure_query, contact_key, contact_key
        )
        
        # First residual connection
        x = self.layer_norm1(pressure_query + attended)
        
        # FFN block
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        x = x.squeeze(1)
        
        # Generate predictions with gating
        gate = self.pressure_gate(x)
        pressure = self.pressure_head(x)
        
        # Apply contact-based gating to pressure
        gated_pressure = F.log_softmax(pressure * gate, dim=-1)
       
        return torch.exp(gated_pressure)
     
class FootFormer(nn.Module):
    def __init__(self, num_joints, joint_dim, pose_embed_dim, num_heads, num_layers, output_dims, seq_len=5, dropout_p=0.1, transformer='transformer', 
                 pos='learnable', mlp_dim=1024, pool='attn',  decoder_dim=1024, mode='pressure', pose_embedder='gcn', pred_distribution=False, contact_conditioned=True):
        super().__init__()
        self.sequence_length = seq_len
        self.pool = pool
        self.mode = mode 
        self.pred_distribution = pred_distribution
       
        self.input_norm = nn.LayerNorm(joint_dim * num_joints)  
        self.pose_embedder = PoseEmbedder(
            num_joints=num_joints, 
            joint_dim=joint_dim, 
            pose_embed_dim=pose_embed_dim, 
            pose_embedder=pose_embedder,
            seq_len=seq_len
        ) 
        self.dropout = nn.Dropout(dropout_p) 
        self.pre_encoder_norm = nn.LayerNorm(pose_embed_dim)
        if transformer == 'multi':
            self.transformer = STT(
                transformer_dim=pose_embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                seq_len=seq_len,
                dropout_p=dropout_p,
                pos=pos,
                mlp_dim=mlp_dim
            )
        elif transformer == 'gru':
            self.transformer = TemporalGRU(
                input_dim=pose_embed_dim,
                hidden_dim=pose_embed_dim,
                n_layers=num_layers
            )
        else:
            self.transformer = Transformer(
                transformer_dim=pose_embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                seq_len=seq_len,
                dropout_p=dropout_p,
                pos=pos,
                mlp_dim=mlp_dim
            )
        self.norm = nn.LayerNorm(pose_embed_dim)
        
        if pool == 'weighted':
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pool == 'attn':
            self.attention_pool = nn.Linear(pose_embed_dim, 1)
        else:
            pool = nn.Identity()
    
        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
       
        self.output_norms = nn.ModuleDict({
            task: nn.LayerNorm(pose_embed_dim) for task in mode
        })
             
        task_activations = {
            'pressure': nn.Softmax(dim=-1) if pred_distribution else nn.Identity(),  
            'contact': nn.Identity(),    # Sigmoid for binary contact
            'com': nn.Identity()        
        }
         
        # Create heads for each task
        self.contact_conditioned = contact_conditioned
        if 'pressure' in mode:
            if contact_conditioned:
                self.task_heads['pressure'] = ContactConditionedPressureHead(
                    input_dim=pose_embed_dim,
                    hidden_dim=decoder_dim,
                    pressure_dim=output_dims['pressure'],
                    contact_dim=output_dims['contact']
                )
            else:
                self.task_heads['pressure'] = self._create_head(
                    pose_embed_dim, 
                    decoder_dim, 
                    output_dims['pressure'],
                    task_activations['pressure']
                )
            
        if 'contact' in mode:
            self.task_heads['contact'] = self._create_head(
                pose_embed_dim,
                decoder_dim,
                output_dims['contact'],
                task_activations['contact']
            )
            
        if 'com' in mode:
            self.task_heads['com'] = self._create_head(
                pose_embed_dim,
                decoder_dim,
                output_dims['com'],
                task_activations['com']
            )
            
    def _create_head(self, input_dim, hidden_dim, output_dim, out_activation=nn.Identity()):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_dim),  
            nn.Linear(hidden_dim, output_dim),
            out_activation
        )
         
    def apply_pool(self, x):
        if self.pool == 'weighted':
            x = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)
        elif self.pool == 'attn':
            attn_weights = self.attention_pool(x)  # [batch, seq_len, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            # Add dropout to attention weights
            attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
            x = torch.bmm(attn_weights.transpose(1,2), x).squeeze(1)
        else:
            x = x[:, x.shape[1]//2, :]
        return x
            
    def forward(self, x):
        """Expect input x to be of shape (batch_size, sequence_length, num_joints, in_channels)"""
        # Embedd pose sequence
        batch_size, seq_len, num_joints, joint_dim = x.shape
        x = x.reshape(batch_size, seq_len, -1)
        x = self.input_norm(x)
        x = x.reshape(batch_size, seq_len, num_joints, joint_dim) 
        
        x = self.pose_embedder(x) 
        x = self.dropout(x)
        x = self.pre_encoder_norm(x)
           
        # Pass through transformer 
        x = self.transformer(x)
        x = self.norm(x)
        
        # Apply pooling
        x = self.apply_pool(x)

        outputs = {}
       
        # Get contact predictions first if using conditioning
        if self.contact_conditioned and 'contact' in self.mode:
            contact_norm = self.output_norms['contact'](x)
            outputs['contact'] = self.task_heads['contact'](contact_norm)
       
        # Handle all tasks
        for task in self.mode:
            if task not in outputs:  # Skip contact if already computed
                task_norm = self.output_norms[task](x)
                if task == 'pressure' and self.contact_conditioned:
                    outputs[task] = self.task_heads[task](task_norm, outputs['contact'])
                else:
                    out = self.task_heads[task](task_norm)
                    if task == 'com':
                        out[..., -1] = F.relu(out[..., -1])
                    outputs[task] = out
        
        return outputs