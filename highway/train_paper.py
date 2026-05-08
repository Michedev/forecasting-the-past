import os
import argparse
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import yaml
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import hydra
import tensorguard as tg
import seaborn as sns
from paper_models import TrajectoryTransformer, EncoderDecoder, TransformerEncoderMLPDecoder

# ==========================================
# Dataset Definition
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, collision_filter=None, max_others=5, pred_len=5, target_relative=False):
        self.pred_len = pred_len
        self.target_relative = target_relative
        self.ego_samples = []
        self.others_samples = []
        self.timesteps = []
        self.has_collisions = []
        
        with h5py.File(h5_path, 'r') as f:
            for ep_key in f.keys():
                if not ep_key.startswith('episode_'): continue
                ep = f[ep_key]
                has_collision = ep.attrs['collision']
                if hasattr(has_collision, 'item'):
                    has_collision = has_collision.item()
                
                if collision_filter is not None and has_collision != collision_filter:
                    continue
                    
                ego = ep['ego_positions'][:]
                if len(ego) <= self.pred_len:
                    continue
                    
                others = ep['other_positions'][:]
                
                others = np.nan_to_num(others)
                if others.shape[1] < max_others:
                    pad = np.zeros((others.shape[0], max_others - others.shape[1], 5))
                    others = np.concatenate([others, pad], axis=1)
                else:
                    others = others[:, :max_others, :]
                    
                others = others.transpose(1, 0, 2)
                
                self.ego_samples.append(ego)
                self.others_samples.append(others)
                self.timesteps.append(torch.arange(len(ego)))
                self.has_collisions.append(has_collision)
                
    def __len__(self):
        return len(self.ego_samples)
        
    def __getitem__(self, idx):
        ego = self.ego_samples[idx]
        others = self.others_samples[idx]
        
        obs_len = len(ego) - self.pred_len * 2
        ego_obs = ego[:obs_len]
        others_obs = others[:, :obs_len, :]
        timesteps_obs = self.timesteps[idx][:obs_len]
        
        tgt_abs = ego[obs_len:obs_len+self.pred_len, :2]
        if self.target_relative:
            tgt = np.diff(tgt_abs, n=1, axis=0)
        else:
            tgt = tgt_abs
        
        return (torch.tensor(ego_obs, dtype=torch.float32), 
                torch.tensor(others_obs, dtype=torch.float32),
                torch.tensor(tgt, dtype=torch.float32),
                torch.tensor(self.has_collisions[idx], dtype=torch.bool),
                timesteps_obs.clone().detach(),
                torch.tensor(tgt_abs, dtype=torch.float32))

def pad_collate(batch):
    egos = [item[0] for item in batch]
    others = [item[1] for item in batch]
    
    lengths = [e.size(0) for e in egos]
    max_len = max(lengths)
    
    padded_egos = []
    padded_others = []
    padding_masks = []
    
    for ego, other, length in zip(egos, others, lengths):
        pad_len = max_len - length
        
        # Pad ego: (seq_len, features) -> (max_len, features)
        padded_egos.append(torch.nn.functional.pad(ego, (0, 0, 0, pad_len)))
        # Pad others: (max_others, seq_len, features) -> (max_others, max_len, features)
        padded_others.append(torch.nn.functional.pad(other, (0, 0, 0, pad_len, 0, 0)))
        
        # Create padding mask: (max_len)
        mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_len > 0:
            mask[-pad_len:] = True
        padding_masks.append(mask)
        
    egos_stacked = torch.stack(padded_egos)
    others_stacked = torch.stack(padded_others)
    masks_stacked = torch.stack(padding_masks)
    
    rest_stacked = []
    for i in range(2, len(batch[0])):
        items = [item[i] for item in batch]
        # If the item is a sequence matching the ego length (like timesteps), pad it
        is_sequence = all(item.dim() > 0 and item.size(0) == length for item, length in zip(items, lengths))
        
        if is_sequence:
            padded_items = []
            for item, length in zip(items, lengths):
                pad_len = max_len - length
                if item.dim() == 1:
                    padded_items.append(torch.nn.functional.pad(item, (0, pad_len)))
                elif item.dim() == 2:
                    padded_items.append(torch.nn.functional.pad(item, (0, 0, 0, pad_len)))
            rest_stacked.append(torch.stack(padded_items))
        else:
            # Check if items are variable length sequences that need checking
            if all(item.dim() > 0 for item in items) and any(item.size(0) != items[0].size(0) for item in items):
                # Find max length for this specific item set
                current_lengths = [item.size(0) for item in items]
                max_item_len = max(current_lengths)
                padded_items = []
                for item, length in zip(items, current_lengths):
                    pad_len = max_item_len - length
                    if item.dim() == 1:
                         # Pad (sequences)
                        padded_items.append(torch.nn.functional.pad(item, (0, pad_len)))
                    elif item.dim() == 2:
                        # Pad (seq_len, dim) -> pad last dim 0, pad 2nd to last dim
                        padded_items.append(torch.nn.functional.pad(item, (0, 0, 0, pad_len)))
                rest_stacked.append(torch.stack(padded_items))
            else:
                rest_stacked.append(torch.stack(items))
            
    return (egos_stacked, others_stacked, *rest_stacked, masks_stacked)

# ==========================================
# Checkpoint Saver
# ==========================================
class TopKCheckpointSaver:
    def __init__(self, k=3, save_dir=""):
        self.k = k
        self.save_dir = save_dir
        self.top_models = []

    def save_checkpoint(self, state_dict, score, epoch):
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(state_dict, checkpoint_path)
        
        self.top_models.append((score, checkpoint_path))
        self.top_models.sort(key=lambda x: x[0], reverse=True)
        
        if len(self.top_models) > self.k:
            _, path_to_remove = self.top_models.pop()
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

# ==========================================
# Training & Evaluation
# ==========================================
def evaluate(model, dataloader, criterion, device, model_type='transformer', X=5, epoch=0, out_dir="", target_relative=False):
    model.eval()
    
    all_gt = []
    all_scores_recon = []
    all_scores_latent = []
    all_scores_last = []
    
    for batch in dataloader:
        tg.clear_dims()
        
        ego_obs, others_obs, tgt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        has_collision = batch[3].to(device)
        timesteps_obs = batch[4].to(device)
        # tgt_abs = batch[5].to(device)
        padding_mask = batch[-1].to(device) if len(batch) > 6 else None
        
        tg.guard(ego_obs, "B, L, F_ego")
        tg.guard(others_obs, "B, M, L, F_other")
        tg.guard(tgt, "B, P, 2")
        tg.guard(has_collision, "B")
        tg.guard(timesteps_obs, "B, L")
        # tg.guard(tgt_abs, "B, P, 2")
        if padding_mask is not None:
            tg.guard(padding_mask, "B, L")
        # print(tg.get_dims())
        if tg.get_dim('L') == 0:
            continue
        # Measure the anomaly scores
        score_recon, score_latent, score_last = model.score(ego_obs, others_obs, tgt, timesteps_obs, padding_mask=padding_mask)
                
        all_gt.extend(has_collision.cpu().numpy())
        all_scores_recon.extend(score_recon.cpu().numpy())
        all_scores_latent.extend(score_latent.cpu().numpy())
        all_scores_last.extend(score_last.cpu().numpy())
            
    # Calculate Metrics
    all_gt = np.array(all_gt, dtype=bool)
    all_scores_recon = np.array(all_scores_recon)
    all_scores_latent = np.array(all_scores_latent)
    all_scores_last = np.array(all_scores_last)

    def safe_auc(gt, scores):
        try:
            return roc_auc_score(gt, scores)
        except ValueError:
            return float('nan')
            
    auc_recon = safe_auc(all_gt, all_scores_recon)
    auc_latent = safe_auc(all_gt, all_scores_latent)
    auc_last = safe_auc(all_gt, all_scores_last)
    
    print(f"  -> Collision Metrics (next {X} steps):")
    print(f"     AUC-ROC (Recon):   {auc_recon:.4f}" if not np.isnan(auc_recon) else "     AUC-ROC (Recon):   N/A")
    print(f"     AUC-ROC (Latent):  {auc_latent:.4f}" if not np.isnan(auc_latent) else "     AUC-ROC (Latent):  N/A")
    print(f"     AUC-ROC (Last):    {auc_last:.4f}" if not np.isnan(auc_last) else "     AUC-ROC (Last):    N/A")
        
    # Plot KDEs
    if out_dir:
        id_mask = ~all_gt
        ood_mask = all_gt
        
        def plot_kde(scores, score_name):
            plt.style.use('plot_style.mplstyle')
            plt.figure()
            if id_mask.any():
                sns.kdeplot(scores[id_mask], label='ID (Safe)', fill=True, alpha=0.5)
            if ood_mask.any():
                sns.kdeplot(scores[ood_mask], label='OOD (Collision)', fill=True, alpha=0.5)
            # plt.title(f'{score_name} KDE Plot')
            plt.xlabel('Score')
            plt.ylabel('Density')
            plt.yticks([])
            plt.xticks([])
            plt.legend()
            plt.tight_layout()
            filename = f'kde_{score_name.lower().replace(" ", "_")}_epoch_{epoch}.png'
            plt.savefig(os.path.join(out_dir, filename))
            filename_pdf = f'kde_{score_name.lower().replace(" ", "_")}_epoch_{epoch}.pdf'
            plt.savefig(os.path.join(out_dir, filename_pdf))
            plt.close()

        plot_kde(all_scores_recon, 'Reconstruction Error')
        plot_kde(all_scores_latent, 'Latent Gradient Norm')
        plot_kde(all_scores_last, 'Last Layer Gradient Norm')

    auc_recon_val = auc_recon if not np.isnan(auc_recon) else 0.0
    auc_latent_val = auc_latent if not np.isnan(auc_latent) else 0.0
    auc_last_val = auc_last if not np.isnan(auc_last) else 0.0
    return auc_recon_val, auc_latent_val, auc_last_val, np.mean(all_scores_recon) if len(all_scores_recon) > 0 else 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Train trajectory prediction model")
    parser.add_argument('--h5_path_train', '--train-path', type=str, default="collision_data/roundabout-v0_train.h5", dest="h5_path_train")
    parser.add_argument('--h5_path_test', '--test-path', type=str, default="collision_data/roundabout-v0_test.h5", dest='h5_path_test')

    parser.add_argument('--obs_len', type=int, default=15)
    parser.add_argument('--pred_len', type=int, default=15)
    parser.add_argument('--max_others', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_type', type=str, choices=['transformer', 'encoder_mlp'], default='encoder_mlp')
    parser.add_argument('--X', type=int, default=5, help='Number of timesteps to check for collision in evaluation')
    parser.add_argument('--test_frequency', type=int, default=2, help='Evaluate model every N epochs')
    parser.add_argument('--target_relative', '--target-relative', action='store_true', help='Compute relative difference for tgt instead of absolute coordinates')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory and save config
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", args.model_type, now)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
        
    if not os.path.exists(args.h5_path_train):
        print(f"Dataset not found at {args.h5_path_train}")
        return
    if not os.path.exists(args.h5_path_test):
        print(f"Dataset not found at {args.h5_path_test}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    # Train only on data WITHOUT collision
    train_dataset = TrajectoryDataset(args.h5_path_train, collision_filter=False, pred_len=args.X, max_others=args.max_others, target_relative=args.target_relative)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)

    print('=' * 20)
    print('Number of train samples:', len(train_dataset))

    # Mixed dataset for periodic evaluation (full trajectories, batch_size=1)
    test_dataset = TrajectoryDataset(args.h5_path_test, collision_filter=None, pred_len=args.X, max_others=args.max_others, target_relative=args.target_relative)
    test_mixed_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)

    print('=' * 20)
    print('Number of test samples:', len(test_dataset))
    print('=' * 20)

    # 2. Initialize Model
    if args.model_type == 'transformer':
        model = TrajectoryTransformer().to(device)
    elif args.model_type == 'encoder_mlp':
        model = TransformerEncoderMLPDecoder(output_dim=2, pred_len=args.X + (-1 if args.target_relative else 0)).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    checkpoint_saver = TopKCheckpointSaver(k=3, save_dir=out_dir)
    auc_log = {}

    # 3. Train Model
    print(f"Training on {len(train_dataset)} safe samples...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            tg.clear_dims()
            
            ego_obs, others_obs, tgt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            timesteps_obs = batch[4].to(device)
            padding_mask = batch[-1].to(device)

            # if random.random() < 0.5 and ego_obs.shape[1] > (10+args.X):
            #     k = random.randint(10, ego_obs.shape[1]-args.X)
            #     ego_obs = ego_obs[:, :k]
            #     tgt = ego_obs[:, k:k+args.X]
            #     others_obs = others_obs[:, :, :k]
            #     padding_mask = padding_mask[:, :k]
            #     timesteps_obs = timesteps_obs[:, :k]

            #     if tgt.shape[1] == 0:
            #         continue


            tg.guard(ego_obs, "B, L, F_ego")
            tg.guard(others_obs, "B, M, L, F_other")
            tg.guard(tgt, "B, P, 2")
            tg.guard(timesteps_obs, "B, L")
            tg.guard(padding_mask, "B, L")
            
            # Teacher forcing: input to decoder is the shifted target sequence
            # Start token is the last observation point
            tgt_input = torch.cat([ego_obs[:, -1:, :2], tgt[:, :-1, :]], dim=1)
            
            optimizer.zero_grad()
            
            if args.model_type == 'transformer':
                out = model(ego_obs, others_obs, tgt_input, timesteps_obs, padding_mask=padding_mask)
            else:
                out = model(ego_obs, others_obs, timesteps_obs, padding_mask=padding_mask)
                
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Periodic Evaluation
        if (epoch + 1) % args.test_frequency == 0:
            print(f"\n--- Evaluation at Epoch {epoch+1} --")
            auc_recon, auc_latent, auc_last, mixed_loss = evaluate(model, test_mixed_loader, criterion, device, args.model_type, args.X, epoch=epoch+1, out_dir=out_dir, target_relative=args.target_relative)
            print(f"Mixed Test Loss: {mixed_loss:.4f}")
            print("-----------------------------------\n")
            
            # Log AUC-ROC to YAML
            auc_log[epoch + 1] = {
                'recon': float(auc_recon),
                'latent': float(auc_latent),
                'last': float(auc_last)
            }
            with open(os.path.join(out_dir, "auc_roc_log.yaml"), "w") as f:
                yaml.dump(auc_log, f)
            
            # Save checkpoint and keep only top 3 based on AUC
            checkpoint_saver.save_checkpoint(model.state_dict(), auc_recon, epoch+1)


if __name__ == "__main__":
    main()
