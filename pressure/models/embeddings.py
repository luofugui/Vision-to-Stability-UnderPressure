import torch.nn as nn   
from pressure.models.gcn import GraphConvolution

class  CNNType(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(CNNType, self).__init__()
        
        # Two parallel convolution paths
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Batch normalization for each path
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        
        # First convolution path
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.leaky_relu(out1)
        out1 = self.dropout(out1)
        
        # Second convolution path
        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.leaky_relu(out2)
        out2 = self.dropout(out2)
        
        # Combine both paths
        out = out1 + out2
        return out
    
class PoseEmbedder(nn.Module):
    def __init__(self, num_joints, joint_dim, pose_embed_dim, seq_len=13, pose_embedder='linear'):
        super().__init__()
        self.pose_embedder = pose_embedder
        input_dim = num_joints * joint_dim
        
        if pose_embedder == 'gcn':
            self.embedding = GraphConvolution(
                in_features=input_dim,
                out_features=pose_embed_dim,
                node_n=seq_len,
                bias=True
            )
        elif pose_embedder == 'cnn':
            self.embedding = CNNType(
                            in_channels=input_dim,
                            out_channels=pose_embed_dim
                        )
        elif pose_embedder == 'linear':
            self.embedding = nn.Linear(input_dim, pose_embed_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_joints, joint_dim)
        """
        batch_size, seq_len, num_joints, joint_dim = x.shape
    
        if self.pose_embedder == 'gcn' or self.pose_embedder == 'linear': 
            x = x.reshape(batch_size, seq_len, num_joints * joint_dim) # -> [batch_size, seq_len, num_joints * joint_dim]
            x = self.embedding(x)  # [batch_size, seq_len, pose_embed_dim]
        elif self.pose_embedder == 'cnn':
            # Reshape for CNN: (batch_size, features, sequence_length)
            x = x.reshape(batch_size, seq_len, num_joints * joint_dim) # -> [batch_size, seq_len, num_joints * joint_dim]
            x = x.transpose(1, 2)  # -> [batch_size, num_joints * joint_dim, seq_len]
            x = self.embedding(x)  # -> [batch_size, pose_embed_dim, seq_len]
            x = x.transpose(1, 2)  # -> [batch_size, seq_len, pose_embed_dim]
            
        else:
            raise ValueError(f"Unknown pose embedder: {self.pose_embedder}")
    
        return x