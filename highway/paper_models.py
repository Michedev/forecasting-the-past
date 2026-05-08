import torch
from torch import nn
import math

def extract_features(ego, others):
    """
    Extracts delta difference using torch.diff.
    :param ego: tensor with shape [batch_size, num_steps, 2]
    :param others: tensor with shape [batch_size, num_agents, num_steps, 2]
    :return: ego_feat, others_feat
    """
    ego_feat = torch.diff(ego, dim=1, prepend=ego[:, 0:1, :])
    others_feat = torch.diff(others, dim=2, prepend=others[:, :, 0:1, :])
    others_feat = torch.cat([others_feat, (others - ego.unsqueeze(1))], dim=1)
    
    return ego_feat, others_feat

def compute_batch_hessian(loss, activation, batch_size):
    """
    Computes the exact Hessian matrix of the loss with respect to the activation.
    Returns a tensor of shape (B, N, N) where N is the flattened feature dimension.
    """
    grad1 = torch.autograd.grad(loss, activation, create_graph=True, retain_graph=True)[0]
    grad1_flat = grad1.view(batch_size, -1)
    B, N = grad1_flat.shape
    hessian = torch.zeros(B, N, N, device=activation.device)
    for i in range(N):
        grad2 = torch.autograd.grad(grad1_flat[:, i].sum(), activation, retain_graph=True)[0]
        hessian[:, i, :] = grad2.view(B, -1)
    return hessian

# ==========================================
# Model Definition
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TrajectoryTransformer(nn.Module):
    def __init__(self, output_dim=2, d_model=64, nhead=4, num_layers=2, timestep_embeddings: int = 100):
        super().__init__()
        self.src_emb = nn.LazyLinear(d_model)
        self.tgt_emb = nn.Linear(output_dim, d_model)
        self.ts_emb = nn.Embedding(timestep_embeddings, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, ego, others, tgt, timesteps, padding_mask=None):
        """
        :param ego: tensor with shape [batch_size, src_steps, 2]
        :param others: tensor with shape [batch_size, num_agents, src_steps, 2]
        :param tgt: tensor with shape [batch_size, tgt_steps, output_dim]
        """
        ego_feat, others_feat = extract_features(ego, others)
        src_emb = self.pos_encoder(self.src_emb(ego_feat) + self.ts_emb(timesteps))        
        tgt_emb = self.pos_encoder(self.tgt_emb(tgt))
        # Causal mask to prevent attending to future tokens in the decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory = self.encoder(src_emb, src_key_padding_mask=padding_mask)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        return self.fc_out(out)

    def score(self, ego, others, tgt, timesteps, padding_mask=None):
        with torch.enable_grad():
            tgt_input = torch.cat([ego[:, -1:, :2], tgt[:, :-1, :2]], dim=1)
            ego_feat, others_feat = extract_features(ego, others)
            src_emb = self.pos_encoder(self.src_emb(ego_feat) + self.ts_emb(timesteps))
            tgt_emb = self.pos_encoder(self.tgt_emb(tgt_input))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)
            memory = self.encoder(src_emb, src_key_padding_mask=padding_mask)
            memory.retain_grad() # Latent space
            dec_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            dec_out.retain_grad() # Last layer input
            preds = self.fc_out(dec_out)
            
            # 1) Reconstruction error
            recon_error = nn.functional.mse_loss(preds, tgt, reduction='none').mean(dim=(1, 2))
            loss = recon_error.sum()
            self.zero_grad()
            loss.backward(retain_graph=True)
            # 2) Gradient w.r.t latent space
            grad_latent = memory.grad
            norm_grad_latent = torch.norm(grad_latent.reshape(grad_latent.size(0), -1), dim=1)
            # 3) Gradient w.r.t last layer input
            grad_last = dec_out.grad
            norm_grad_last = torch.norm(grad_last.reshape(grad_last.size(0), -1), dim=1)
        return recon_error.detach(), norm_grad_latent.detach(), norm_grad_last.detach()    

    def hessian_score(self, ego, others, tgt, timesteps, padding_mask=None):
        with torch.enable_grad():
            tgt_input = torch.cat([ego[:, -1:, :2], tgt[:, :-1, :2]], dim=1)
            ego_feat, others_feat = extract_features(ego, others)
            src_emb = self.pos_encoder(self.src_emb(ego_feat) + self.ts_emb(timesteps))
            tgt_emb = self.pos_encoder(self.tgt_emb(tgt_input))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)
            
            memory = self.encoder(src_emb, src_key_padding_mask=padding_mask)
            dec_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            preds = self.fc_out(dec_out)
            
            # 1) Reconstruction error
            recon_error = nn.functional.mse_loss(preds, tgt, reduction='none').mean(dim=(1, 2))
            loss = recon_error.sum()
            
            B = ego.size(0)
            # 2) Hessian w.r.t latent space
            hessian_latent = compute_batch_hessian(loss, memory, B)
            # 3) Hessian w.r.t last layer input
            hessian_last = compute_batch_hessian(loss, dec_out, B)
            
        return recon_error.detach(), hessian_latent.detach(), hessian_last.detach()

class TransformerEncoderMLPDecoder(nn.Module):
    def __init__(self, output_dim=2, pred_len=15, d_model=64, nhead=4, num_layers=5, timestep_embeddings: int = 200, mlp_hidden_dim: int = 128, mlp_num_layers: int = 2):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        self.src_emb = nn.LazyLinear(d_model)
        self.others_proj = nn.LazyLinear(d_model)
        self.ts_emb = nn.Embedding(timestep_embeddings, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        layers = []
        if mlp_num_layers == 1:
            layers.append(nn.Linear(d_model, pred_len * output_dim))
        else:
            layers.append(nn.Linear(d_model, mlp_hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(mlp_num_layers - 2):
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(mlp_hidden_dim, pred_len * output_dim))
            
        self.mlp_decoder = nn.Sequential(*layers)
        
    def forward(self, ego, others, timesteps, padding_mask=None):
        """
        :param ego: tensor with shape [batch_size, src_steps, 2]
        :param others: tensor with shape [batch_size, num_agents, src_steps, 2]
        """
        ego_feat, others_feat = extract_features(ego, others)
        src_emb = self.pos_encoder(self.src_emb(ego_feat) + self.ts_emb(timesteps))
        memory = self.encoder(src_emb, src_key_padding_mask=padding_mask)

        B, T, _ = memory.shape
        M = others_feat.shape[1]
        others_emb = self.others_proj(others_feat)
        memory_flat = memory.reshape(B * T, 1, -1)
        others_emb_flat = others_emb.permute(0, 2, 1, 3).reshape(B * T, M, -1)
        msg, _ = self.fusion_attn(query=memory_flat, key=others_emb_flat, value=others_emb_flat)
        msg = msg.view(B, T, -1)
        memory = memory + msg
        
        # Use the last valid hidden state of the encoder to predict the future trajectory
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1)
            last_hidden = memory[torch.arange(memory.size(0)), lengths - 1, :]
        else:
            last_hidden = memory[:, -1, :]
        
        out = self.mlp_decoder(last_hidden)
        return out.view(-1, self.pred_len, self.output_dim)

    def score(self, ego, others, tgt, timesteps, padding_mask=None):
        with torch.enable_grad():
            ego_feat, others_feat = extract_features(ego, others)
            src_emb = self.pos_encoder(self.src_emb(ego_feat) + self.ts_emb(timesteps))
            memory = self.encoder(src_emb, src_key_padding_mask=padding_mask)

            B, T, _ = memory.shape
            M = others_feat.shape[1]
            others_emb = self.others_proj(others_feat)
            memory_flat = memory.reshape(B * T, 1, -1)
            others_emb_flat = others_emb.permute(0, 2, 1, 3).reshape(B * T, M, -1)
            msg, _ = self.fusion_attn(query=memory_flat, key=others_emb_flat, value=others_emb_flat)
            msg = msg.view(B, T, -1)
            memory = memory + msg
            memory.retain_grad() # Latent space
            
            if padding_mask is not None:
                lengths = (~padding_mask).sum(dim=1)
                last_hidden = memory[torch.arange(memory.size(0)), lengths - 1, :]
            else:
                last_hidden = memory[:, -1, :]
                
            x = last_hidden
            if len(self.mlp_decoder) > 1:
                for i in range(len(self.mlp_decoder) - 1):
                    x = self.mlp_decoder[i](x)
            x.retain_grad() # Last layer input
            out = self.mlp_decoder[-1](x)
            
            preds = out.view(-1, self.pred_len, self.output_dim)
            min_len = min(preds.size(1), tgt.size(1))
            preds_c = preds[:, :min_len, :]
            tgt_c = tgt[:, :min_len, :]
            
            # 1) Reconstruction error
            recon_error = nn.functional.mse_loss(preds_c, tgt_c, reduction='none').mean(dim=(1, 2))
            loss = recon_error.sum()
            self.zero_grad()
            loss.backward(retain_graph=True)

            # 2) Gradient w.r.t latent space
            grad_latent = memory.grad
            norm_grad_latent = torch.norm(grad_latent.reshape(grad_latent.size(0), -1), dim=1)
            # 3) Gradient w.r.t last layer input
            grad_last = x.grad
            norm_grad_last = torch.norm(grad_last.reshape(grad_last.size(0), -1), dim=1)
        return recon_error.detach(), norm_grad_latent.detach(), norm_grad_last.detach()

    def hessian_score(self, ego, others, tgt, timesteps, padding_mask=None):
        with torch.enable_grad():
            ego_feat, others_feat = extract_features(ego, others)
            src_emb = self.pos_encoder(self.src_emb(ego_feat) + self.ts_emb(timesteps))
            memory = self.encoder(src_emb, src_key_padding_mask=padding_mask)

            B, T, _ = memory.shape
            M = others_feat.shape[1]
            others_emb = self.others_proj(others_feat)
            memory_flat = memory.reshape(B * T, 1, -1)
            others_emb_flat = others_emb.permute(0, 2, 1, 3).reshape(B * T, M, -1)
            msg, _ = self.fusion_attn(query=memory_flat, key=others_emb_flat, value=others_emb_flat)
            msg = msg.view(B, T, -1)
            memory = memory + msg
            
            if padding_mask is not None:
                lengths = (~padding_mask).sum(dim=1)
                last_hidden = memory[torch.arange(memory.size(0)), lengths - 1, :]
            else:
                last_hidden = memory[:, -1, :]
                
            x = last_hidden
            if len(self.mlp_decoder) > 1:
                for i in range(len(self.mlp_decoder) - 1):
                    x = self.mlp_decoder[i](x)
            out = self.mlp_decoder[-1](x)
            
            preds = out.view(-1, self.pred_len, self.output_dim)
            min_len = min(preds.size(1), tgt.size(1))
            preds_c = preds[:, :min_len, :]
            tgt_c = tgt[:, :min_len, :]
            
            # 1) Reconstruction error
            recon_error = nn.functional.mse_loss(preds_c, tgt_c, reduction='none').mean(dim=(1, 2))
            loss = recon_error.sum()
            
            # 2) Hessian w.r.t latent space
            hessian_latent = compute_batch_hessian(loss, memory, B)
            # 3) Hessian w.r.t last layer input
            hessian_last = compute_batch_hessian(loss, x, B)
            
        return recon_error.detach(), hessian_latent.detach(), hessian_last.detach()