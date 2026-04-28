from itertools import chain
from beartype import beartype
import numpy as np
import pytorch_lightning as pl
from typing import Literal
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from joodu.metrics.f1 import compute_f1_at_retention, compute_f1_auc
from joodu.metrics.fpr import compute_fpr_at_retention, compute_ap_at_retention
from joodu.model.losses import GaussianMixtureNLLLoss
from joodu.metrics import ade, fde, wade, wfde, mr
from joodu.metrics.uncertainty import compute_uncertainty_from_trajectories
from torchmetrics import AUROC, MeanMetric
import tensorguard as tg

from joodu.model.trajectory_utils import rotate_data_
from joodu.model.utils import load_ood_model
from joodu.utils import fix_trajectory
from utils import TemporalData, rotate_trajectory


def _interpolate_x( vector, target_size, mode='linear'):
    vector= vector.permute(0, 2, 1)

    vector = torch.nn.functional.interpolate(vector, target_size, mode=mode)
    
    vector= vector.permute(0, 2, 1)

    return vector

def interpolate_x(batch, perc_x=0.5):
    orig_x = batch.x.clone()  #[b,25,2]
    orig_steps_x = batch.x.shape[1]
    orig_steps_y = batch.y.shape[1]
    perc_y = 1 - perc_x
    new_steps_x = int(perc_x * orig_steps_x)
    new_steps_y = orig_steps_x - new_steps_x
    new_x = orig_x[:, :new_steps_x]
    new_y = orig_x[:, new_steps_x:]

    new_vel_x = batch.velocities[:, 0:new_steps_x]
    new_vel_y = batch.velocities[:, new_steps_x:(new_steps_x+new_steps_y)]
    new_acc_x = batch.accelerations[:, 0:new_steps_x]
    new_acc_y = batch.accelerations[:, new_steps_x:(new_steps_x+new_steps_y)]

    
    new_x = _interpolate_x(new_x, orig_steps_x, 'linear')
    new_y  = _interpolate_x(new_y, orig_steps_y, 'linear')

    new_vel_x = _interpolate_x(new_vel_x, orig_steps_x, 'linear')
    new_vel_y = _interpolate_x(new_vel_y, orig_steps_y, 'linear')

    new_acc_x = _interpolate_x(new_acc_x, orig_steps_x, 'linear')
    new_acc_y = _interpolate_x(new_acc_y, orig_steps_y, 'linear')

    new_x = new_x - new_y[:, :1]
    new_y = new_y - new_y[:, :1]
    batch.x = new_x
    batch.y = new_y
    batch.y = torch.bmm(batch.y, batch.rotate_mat)
    batch.velocities = torch.cat(tensors=(new_vel_x, new_vel_y), dim=1)
    batch.accelerations = torch.cat((new_acc_x, new_acc_y), dim=1)

    
    reg_mask = (~batch['padding_mask'][:, :orig_steps_x]).float()
    reg_mask = reg_mask.unsqueeze(1)  # Add channel dimension: [batch_size, 1, num_steps]
    reg_mask = torch.nn.functional.interpolate(reg_mask, size=orig_steps_x + orig_steps_y, mode='linear')
    reg_mask = reg_mask.squeeze(1)  # Remove channel dimension: [batch_size, num_steps]
    reg_mask = reg_mask > 0
    batch.padding_mask = ~reg_mask

    return batch



class TrajPredSSDecoder(pl.LightningModule):

    @beartype
    def __init__(self,
                 local_encoder,
                 global_interactor,
                 decoder,
                 ae_decoder,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-6,
                 ae_loss_weight: float = 0.1,
                 num_modes: int = 5,
                 historical_steps: int = 25,
                 future_steps: int = 25,
                 perc_x: float = 0.5,
                 gradient_type: Literal['latent', 'all', 'last'] = 'latent',):
        super().__init__()
        assert 0.0 <= perc_x <= 1.0
        self.save_hyperparameters(ignore=['local_encoder', 'global_interactor', 'decoder', 'ae_decoder'])
        
        # Model components
        self.local_encoder = local_encoder
        self.global_interactor = global_interactor
        self.decoder = decoder
        self.ae_decoder = ae_decoder
        
        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.ae_loss_weight = ae_loss_weight
        self.num_modes = num_modes
        self.historical_steps = historical_steps
        self.future_steps: int = future_steps
        self.perc_x = perc_x
        
        # Loss functions
        self.trajectory_loss = GaussianMixtureNLLLoss(reduction="none")
        self.reconstruction_loss = torch.nn.MSELoss(reduction='none')

        self.true_ood_model = load_ood_model()
        self.gradient_type = gradient_type
        
        # Metrics
        self.minADE = ade.MinADE()
        self.minFDE = fde.MinFDE()
        self.wADE = wade.WADE()
        self.wFDE = wfde.WFDE()
        self.MR = mr.MR()
        self.minADE_no_opt = ade.MinADE()
        self.minFDE_no_opt = fde.MinFDE()
        self.wADE_no_opt = wade.WADE()
        self.wFDE_no_opt = wfde.WFDE()
        self.MR_no_opt = mr.MR()


        self.aucroc_ood = AUROC(2)
        self.aucroc_ood_no_opt = AUROC(2)
        self.avg_uc_full = MeanMetric()
        self.avg_uc_id = MeanMetric()
        self.avg_uc_ood = MeanMetric()

    def forward(self, data, return_embeds=False):
            local_embed = self.local_encoder(data)
            global_embed = self.global_interactor(data=data, local_embed=local_embed)

            loc_scale, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
            loc, scale = loc_scale.chunk(2, dim=-1)
            pi = pi.transpose(0, 1)
            if return_embeds:
                return loc, scale, pi, local_embed, global_embed
            return loc, scale, pi

    @classmethod
    def join_latent_vectors(cls, local_vector, global_vector):
        """Joins local and global latent vectors, repeating local vector for each mode."""
        tg.guard(local_vector, "*, LD")
        tg.guard(global_vector, "NM, *, GD")
        global_vector = global_vector.transpose(0,1)
        joined_vector = torch.cat((global_vector, local_vector.unsqueeze(1).expand(-1, global_vector.shape[1], -1) ), dim=-1)
        # joined_vector = joined_vector.transpose(0, 1)

        tg.guard(joined_vector, "*, NM, GD+LD")
        return joined_vector

    @classmethod
    def unjoin_latent_vectors(cls, latent_vector):
        """Splits the latent vector into local and global components."""
        tg.guard(latent_vector, "*, NM, GD+LD")
        
        global_v, local_v = latent_vector[:, :, :128], latent_vector[:, 0, 128:]
        global_v = global_v.transpose(0, 1)

        return local_v, global_v
    
    
    def training_step(self, batch, batch_idx):
        """Training step with both trajectory prediction and autoencoder reconstruction losses"""
        with torch.no_grad():
            agent_index = batch.agent_index[batch.valid]
            reg_mask = ~batch['padding_mask'][:, self.historical_steps:]
            # reg_mask = reg_mask
            rotate_data_(batch)


            data = interpolate_x(batch, self.perc_x)
            
            # Forward pass through encoder
            local_embed = self.local_encoder(data)
            global_embed = self.global_interactor(data=data, local_embed=local_embed)
            
        # Autoencoder reconstruction loss
        # Use local embeddings for reconstruction
        loc_scale, pi = self.ae_decoder(local_embed, global_embed)
        loc, scale = loc_scale.chunk(2, dim=-1)
        loss = self.trajectory_loss(loc[:, agent_index], pi[agent_index], scale[:, agent_index], batch.y[agent_index], reg_mask[agent_index])
        return loss.mean()
    
    def on_validation_epoch_start(self) -> None:
        self.alpha_hat_opt = []
        self.alpha_hat_no_opt = []
        self.is_ood = []
        return super().on_validation_epoch_start()

    
    def validation_step(self, batch, batch_idx: int) -> None:
        """Validation step for trajectory prediction and OOD detection using trajectory loss"""
        tg.clear_dims()
        batch.x = fix_trajectory(batch.x)
        true_local_embed = self.local_encoder(batch)
        data = batch
        data = batch.to(self.device)
        agent_index = data.agent_index[data.valid]
        tg.set_dim('Ba', len(agent_index))
        rotate_data_(data)
        
        # Forward pass through encoder
        loc, scale, pi, local_embed, global_embed = self.forward(data, return_embeds=True)
        
        # Get agent-specific predictions (NO OPTIMIZATION)
        loc_agent_no_opt = loc[:, agent_index]
        scale_agent_no_opt = scale[:, agent_index]  
        pi_agent_no_opt = pi[:, agent_index]
        
        # Compute NO-OPT metrics
        y_agent = data.y[agent_index]
        y_hat_agent_no_opt = loc_agent_no_opt.transpose(0, 1)  # [agents, modes, future_steps, 2]
        pi_agent_no_opt_softmax = pi_agent_no_opt.transpose(0, 1).softmax(dim=1)  # [agents, modes]
        
        self.minADE_no_opt.update(y_hat_agent_no_opt, y_agent)
        self.minFDE_no_opt.update(y_hat_agent_no_opt, y_agent)
        self.wADE_no_opt.update(y_hat_agent_no_opt, pi_agent_no_opt_softmax, y_agent)
        self.wFDE_no_opt.update(y_hat_agent_no_opt, pi_agent_no_opt_softmax, y_agent)
        self.MR_no_opt.update(y_hat_agent_no_opt, y_agent)

        # Create interpolated version of batch for OOD detection
        data_interpolated = batch.clone()
        data_interpolated: TemporalData = interpolate_x(data_interpolated, self.perc_x)
        
        # Forward pass on interpolated data with gradient computation
        if self.gradient_type == 'latent':
            local_embed, grad, ss_loss = self.compute_gradient_latent(agent_index, data_interpolated)
        elif self.gradient_type == 'last':
            local_embed, grad, ss_loss = self.compute_gradient_last_layer(agent_index, data_interpolated)
        else:
            local_embed, grad = self.compute_gradient_all_layers(agent_index, data_interpolated)
        # Compute various gradient aggregation methods for alpha scores
        grad_agent = grad[agent_index]
        alpha_methods = self.compute_gradient_aggregations(grad_agent)
        # for k in list(alpha_methods.keys()):
        #     print(ss_loss.shape, alpha_methods[k].shape, sep=' - ')
        #     alpha_methods[k+'_with_loss'] = alpha_methods[k] * 10 + 0.1 * (ss_loss.abs())
        # Compute trajectory loss for all agents
        alpha_hat_no_opt = self.true_ood_model.predict_ood_score(true_local_embed.detach().cpu().numpy())
        alpha_hat_no_opt = torch.from_numpy(alpha_hat_no_opt).float().to(local_embed.device)
        alpha_hat_no_opt_agent = alpha_hat_no_opt[agent_index]
        

        # OOD detection using trajectory loss as uncertainty score
        is_ood = data.ood[agent_index]
            
        # Store all alpha scores for different aggregation methods
        for method_name, alpha_score in alpha_methods.items():
            if not hasattr(self, f'alpha_hat_opt_{method_name}'):
                setattr(self, f'alpha_hat_opt_{method_name}', [])
            getattr(self, f'alpha_hat_opt_{method_name}').append(alpha_score.detach().cpu().numpy())
        
        # Store the no-opt baseline
        self.alpha_hat_no_opt.append(alpha_hat_no_opt_agent.detach().cpu().numpy())
        self.is_ood.append(is_ood.detach().cpu().numpy())
        
        # Log metrics
        metrics_dict = {
            "val/minADE_no_opt": self.minADE_no_opt.compute(),
            "val/minFDE_no_opt": self.minFDE_no_opt.compute(),
            "val/wADE_no_opt": self.wADE_no_opt.compute(),
            "val/wFDE_no_opt": self.wFDE_no_opt.compute(),
            "val/MR_no_opt": self.MR_no_opt.compute(),
        }

        # metrics_ood_dict = {
        #     "val/aucroc_ood": self.aucroc_ood.compute(),
        #     'val/aucroc_ood_no_opt': self.aucroc_ood_no_opt.compute()
        # }
        
        self.log_dict(
            metrics_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(agent_index),
            sync_dist=True,
        )

        return 

    def compute_gradient_all_layers(self, agent_index, data_interpolated):
        # Setup gradient collection hooks for linear layers
        linear_layer_grads = []
        hooks = []
        
        def register_grad_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                linear_layer_grads.append(grad_output[0])
        
        # Register hooks on all linear layers
        for module in chain(self.local_encoder.modules(), self.global_interactor.modules(), self.ae_decoder.modules()):
            if isinstance(module, torch.nn.Linear):
                hook = module.register_backward_hook(register_grad_hook)
                hooks.append(hook)
        
        # Enable gradients for model parameters
        for p in chain(self.local_encoder.parameters(), self.global_interactor.parameters(), self.ae_decoder.parameters()):
            p.requires_grad_(True)
        
        # Forward pass on interpolated data
        with torch.set_grad_enabled(True):
            local_embed = self.local_encoder(data_interpolated)
            global_embed = self.global_interactor(data=data_interpolated, local_embed=local_embed)

            loc_scale, pi_interp = self.decoder(local_embed=local_embed, global_embed=global_embed)
            loc_interp, scale_interp = loc_scale.chunk(2, dim=-1)
            pi_interp = pi_interp.transpose(0, 1)
            
            # Compute trajectory loss on interpolated data for all agents
            reg_mask = ~data_interpolated['padding_mask'][:, self.historical_steps:]
            
            # Get agent-specific predictions and ground truth
            agent_loc = loc_interp[:, agent_index]
            agent_scale = scale_interp[:, agent_index]
            agent_pi = pi_interp[:, agent_index]
            agent_y = data_interpolated.y[agent_index]
            agent_mask = reg_mask[agent_index]
            agent_pi = agent_pi.transpose(0,1)
            loss = self.trajectory_loss.forward(agent_loc, agent_pi, agent_scale, agent_y, agent_mask)
            loss = loss.mean()
            loss.backward()
        
        # Remove hooks and disable gradients
        for hook in hooks:
            hook.remove()
        
        for p in chain(self.local_encoder.parameters(), self.global_interactor.parameters(), self.ae_decoder.parameters()):
            p.requires_grad_(False)
        
        # Process collected gradients from linear layers
        grad = []
        for layer_grad in linear_layer_grads:
            if layer_grad is not None:
                # Flatten gradient while preserving batch dimension
                good_grad = len(layer_grad.shape) == 2 and (layer_grad.shape[0] in (len(data_interpolated.x), len(agent_index)))
                good_grad = good_grad or (len(layer_grad.shape) == 3 and (layer_grad.shape[1] in (len(data_interpolated.x), len(agent_index))))
                if not good_grad:
                    continue
                if len(layer_grad.shape) > 2:
                    if layer_grad.shape[1] in (len(data_interpolated.x), len(agent_index)):
                        layer_grad = layer_grad.transpose(0,1)
                    layer_grad = layer_grad.flatten(1)
                grad.append(layer_grad)
        
        # print(f'{data_interpolated.x.shape=}, {len(agent_index)=}')
        # print([el.shape for el in grad])
        grad = torch.cat(grad, dim=-1)
        for p in chain(self.local_encoder.parameters(), self.global_interactor.parameters(), self.ae_decoder.parameters()):
            p.requires_grad_(False)
        return local_embed, grad


    def compute_gradient_latent(self, agent_index, data_interpolated):
        with torch.set_grad_enabled(True):

            local_embed = self.local_encoder(data_interpolated)
            local_embed.requires_grad_(True)  # Enable gradients for local_embed
            global_embed = self.global_interactor(data=data_interpolated, local_embed=local_embed)

            loc_scale, pi_interp = self.decoder(local_embed=local_embed, global_embed=global_embed)
            loc_interp, scale_interp = loc_scale.chunk(2, dim=-1)
            pi_interp = pi_interp.transpose(0, 1)
                
                # Compute trajectory loss on interpolated data for all agents
            reg_mask = ~data_interpolated['padding_mask'][:, self.historical_steps:]
                
                # Get agent-specific predictions and ground truth
            agent_loc = loc_interp[:, agent_index]
            agent_scale = scale_interp[:, agent_index]
            agent_pi = pi_interp[:, agent_index]
            agent_y = data_interpolated.y[agent_index]
            agent_mask = reg_mask[agent_index]
            agent_pi = agent_pi.transpose(0,1)
            ss_loss = self.trajectory_loss.forward(agent_loc, agent_pi, agent_scale, agent_y, agent_mask)
            loss = ss_loss.mean()
                
                # Compute gradients w.r.t. local_embed
            grad = torch.autograd.grad(outputs=loss, inputs=local_embed, 
                                        create_graph=False, retain_graph=False)[0]
                                    
        return local_embed,grad,ss_loss

    def compute_gradient_last_layer(self, agent_index, data_interpolated):
        """Compute gradient w.r.t. the input of the last linear layer in the decoder."""
        last_layer_input = None
        last_layer_grad = None
        
        def capture_last_layer_input(module, input, output):
            nonlocal last_layer_input
            last_layer_input = input[0]
            last_layer_input.requires_grad_(True)
        
        def capture_last_layer_grad(grad):
            nonlocal last_layer_grad
            last_layer_grad = grad
        
        # Find the last linear layer in the decoder
        decoder_layers = list(self.ae_decoder.modules())
        last_linear_layer = None
        for layer in reversed(decoder_layers):
            if isinstance(layer, torch.nn.Linear):
                last_linear_layer = layer
                break
        
        if last_linear_layer is None:
            raise ValueError("No linear layer found in decoder")
        
        # Register forward hook to capture input
        hook = last_linear_layer.register_forward_hook(capture_last_layer_input)
        
        try:
            with torch.set_grad_enabled(True):
                local_embed = self.local_encoder(data_interpolated)
                global_embed = self.global_interactor(data=data_interpolated, local_embed=local_embed)

                loc_scale, pi_interp = self.ae_decoder(local_embed=local_embed, global_embed=global_embed)
                loc_interp, scale_interp = loc_scale.chunk(2, dim=-1)
                pi_interp = pi_interp.transpose(0, 1)
                
                # Compute trajectory loss on interpolated data for all agents
                reg_mask = ~data_interpolated['padding_mask'][:, self.historical_steps:]
                
                # Get agent-specific predictions and ground truth
                agent_loc = loc_interp[:, agent_index]
                agent_scale = scale_interp[:, agent_index]
                agent_pi = pi_interp[:, agent_index]
                agent_y = data_interpolated.y[agent_index]
                agent_mask = reg_mask[agent_index]
                agent_pi = agent_pi.transpose(0,1)
                ss_loss = self.trajectory_loss.forward(agent_loc, agent_pi, agent_scale, agent_y, agent_mask)
                loss = ss_loss.mean()
                
                # Register backward hook on the captured input
                if last_layer_input is not None:
                    last_layer_input.register_hook(capture_last_layer_grad)
                
                last_layer_grad = torch.autograd.grad(loss, last_layer_input)[0]
                last_layer_grad = last_layer_grad.transpose(0, 1).flatten(1)
                print(last_layer_grad.shape)
                
        finally:
            hook.remove()
        
        if last_layer_grad is None:
            raise ValueError("Failed to capture gradient from last layer")
        
        return local_embed, last_layer_grad, ss_loss

    def compute_gradient_aggregations(self, grad_agent):
        """Compute various gradient aggregation methods for alpha scores."""
        alpha_methods = {
            'l1_sum': grad_agent.abs().sum(dim=-1),  # L1 norm (sum of absolute values)
            'l2_norm': grad_agent.norm(dim=-1),  # L2 norm (Euclidean norm)
            'max_abs': grad_agent.abs().max(dim=-1)[0],  # Max absolute value
            'mean_abs': grad_agent.abs().mean(dim=-1),  # Mean absolute value
            'std': grad_agent.std(dim=-1),  # Standard deviation
            'var': grad_agent.var(dim=-1),  # Variance
            'l2_squared': (grad_agent ** 2).sum(dim=-1),  # Sum of squares
            'linf_norm': grad_agent.abs().max(dim=-1)[0],  # L-infinity norm
            'frobenius': grad_agent.norm(p='fro', dim=-1),  # Frobenius norm
            # 'nuclear': torch.linalg.norm(grad_agent, ord='nuc', dim=-1)  # Nuclear norm
        }
        return alpha_methods

    def calc_alpha_score(self, batch, norm_grad: bool = True, agent_only: bool = True, local_embed=None):
        data_interpolated = batch.clone()
        agent_index = batch.agent_index[batch.valid]
        data_interpolated = interpolate_x(data_interpolated, self.perc_x)
        
        # Forward pass on interpolated data with gradient computation
        with torch.set_grad_enabled(True):
            if local_embed is None:
                local_embed = self.local_encoder(data_interpolated)
            local_embed.requires_grad_(True)  # Enable gradients for local_embed
            global_embed = self.global_interactor(data=data_interpolated, local_embed=local_embed)

            loc_scale, pi_interp = self.decoder(local_embed=local_embed, global_embed=global_embed)
            loc_interp, scale_interp = loc_scale.chunk(2, dim=-1)
            pi_interp = pi_interp.transpose(0, 1)
            
            # Compute trajectory loss on interpolated data for all agents
            reg_mask = ~data_interpolated['padding_mask'][:, self.historical_steps:]
            
            # Get agent-specific predictions and ground truth
            if agent_only:
                agent_loc = loc_interp[:, agent_index]
                agent_scale = scale_interp[:, agent_index]
                agent_pi = pi_interp[:, agent_index]
                agent_y = data_interpolated.y[agent_index]
                agent_mask = reg_mask[agent_index]
            else:
                agent_loc = loc_interp
                agent_scale = scale_interp
                agent_pi = pi_interp
                agent_y = data_interpolated.y
                agent_mask = reg_mask


            agent_pi = agent_pi.transpose(0,1)

            loss = self.trajectory_loss.forward(agent_loc, agent_pi, agent_scale, agent_y, agent_mask)
            loss = loss.mean()
            
            # Compute gradients w.r.t. local_embed
            grad = torch.autograd.grad(outputs=loss, inputs=local_embed, 
                                     create_graph=False, retain_graph=False)[0]
        if agent_only:
            grad_agent = grad[agent_index]
        else:
            grad_agent = grad
        
        # Use l2_squared aggregation method like in validation_step
        if norm_grad:
            return grad_agent.abs().amax(dim=-1)
        else:
            return grad_agent

    def on_validation_epoch_end(self):
        """Reset metrics at the end of validation epoch."""
        self.minADE.reset()
        self.minFDE.reset()
        self.wADE.reset()
        self.wFDE.reset()
        self.MR.reset()
        self.aucroc_ood.reset()
        self.avg_uc_full.reset()
        self.avg_uc_id.reset()
        self.avg_uc_ood.reset()
        
        if hasattr(self, 'is_ood') and len(self.is_ood) > 0:
            is_ood = np.concatenate(self.is_ood, axis=0)
            alpha_hat_no_opt = np.concatenate(self.alpha_hat_no_opt, axis=0)
            
            # Compute AUC for no-opt baseline
            auc_score_no_opt = roc_auc_score(is_ood, alpha_hat_no_opt).item()
            f1_auc_no_opt = compute_f1_auc(is_ood, alpha_hat_no_opt)
            f1_95_no_opt = compute_f1_at_retention(is_ood, y_scores=alpha_hat_no_opt)
            fpr_95_no_opt = compute_fpr_at_retention(is_ood, y_scores=alpha_hat_no_opt)
            ap_95_no_opt = compute_ap_at_retention(is_ood, y_scores=alpha_hat_no_opt)

            self.log('val/aucroc_ood_no_opt', auc_score_no_opt, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val/f1_auc_no_opt', value=f1_auc_no_opt, prog_bar=False, on_epoch=True)
            self.log('val/f1_95_no_opt', value=f1_95_no_opt, prog_bar=False, on_epoch=True)
            self.log('val/fpr_95_no_opt', value=fpr_95_no_opt, prog_bar=False, on_epoch=True)
            self.log('val/ap_95_no_opt', value=ap_95_no_opt, prog_bar=False, on_epoch=True)
            
            # Dynamically detect all gradient aggregation methods
            alpha_hat_attrs = [attr for attr in dir(self) if attr.startswith('alpha_hat_opt_')]
            
            for attr_name in alpha_hat_attrs:
                alpha_scores = getattr(self, attr_name)
                if isinstance(alpha_scores, list) and len(alpha_scores) > 0:
                    alpha_hat = np.concatenate(alpha_scores, axis=0)
                    # print(f'{alpha_hat.shape=} {alpha_hat=}')
                    auc_score = roc_auc_score(is_ood, alpha_hat).item()
                    method_name = attr_name.replace('alpha_hat_opt_', '')
                    f1_auc = compute_f1_auc(is_ood, alpha_hat)
                    f1_95 = compute_f1_at_retention(is_ood, alpha_hat)
                    fpr_95 = compute_fpr_at_retention(is_ood, alpha_hat)
                    ap_95 = compute_ap_at_retention(is_ood, alpha_hat)

                    self.log(f'val/aucroc_ood_{method_name}', value=auc_score, prog_bar=False, on_step=False, on_epoch=True)
                    self.log(f'val/f1_auc_{method_name}', value=f1_auc, prog_bar=False, on_epoch=True)
                    self.log(f'val/f1_95_{method_name}', value=f1_95, prog_bar=False, on_epoch=True)
                    self.log(f'val/fpr_95_{method_name}', value=fpr_95, prog_bar=False, on_epoch=True)
                    self.log(f'val/ap_95_{method_name}', value=ap_95, prog_bar=False, on_epoch=True)
                    # Reset for next epoch
                    setattr(self, attr_name, [])
            
            # Log the best performing method in progress bar
            # Use l1_sum as default for progress bar
            if hasattr(self, 'alpha_hat_opt_l1_sum') and len(self.alpha_hat_opt_l1_sum) > 0:
                alpha_hat_l1 = np.concatenate(self.alpha_hat_opt_l1_sum, axis=0)
                auc_score_l1 = roc_auc_score(is_ood, alpha_hat_l1).item()
                self.log('val/aucroc_ood', auc_score_l1, prog_bar=True, on_step=False, on_epoch=True)
            
            # Reset lists
            self.alpha_hat_no_opt = []
            self.is_ood = []

    def configure_optimizers(self):
        """Configure optimizer for training."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
