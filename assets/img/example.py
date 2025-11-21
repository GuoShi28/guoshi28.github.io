import math

class SimpleMHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape

        Q = self.W_q(x)  # (B, L, D)
        K = self.W_k(x)  # (B, L, D)
        V = self.W_v(x)  # (B, L, D)

        # ========== TODO 1: 把 Q/K/V reshape 成多头形式 ==========
        Q = ...
        K = ...
        V = ...
        
        # ========== TODO 2: 计算注意力 ==========
        # 注意力分数: (B, H, L, head_dim) @ (B, H, head_dim, L) -> (B, H, L, L)
        scores = ....

        attn_heads = torch.matmul(attn_weights, V)   
        # todo:合并多头: (B, H, L, head_dim) -> (B, L, D)
        
        out = self.W_o(attn)  # (B, L, D)
        return out
