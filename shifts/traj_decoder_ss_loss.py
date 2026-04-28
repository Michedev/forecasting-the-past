from shifts.traj_decoder_ss import TrajPredSSDecoder, interpolate_x


class TrajPredSSDecoderLoss(TrajPredSSDecoder):
    """
    Uses loss function instead of gradient for OOD detection
    """


    def compute_gradient_aggregations(self, loss_fn):# -> dict[str, Any]:
        """Compute various gradient aggregation methods for alpha scores."""
        return dict(alpha_hat_opt_loss=loss_fn)

    def calc_alpha_score(self, batch):
        data_interpolated = batch.clone()
        agent_index = batch.agent_index[batch.valid]
        data_interpolated = interpolate_x(data_interpolated, self.perc_x)

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
        trajectory_loss_reduction = None
        if self.trajectory_loss.reduction != 'none':
            trajectory_loss_reduction = self.trajectory_loss.reduction
            self.trajectory_loss.reduction = 'none'
        loss = self.trajectory_loss.forward(agent_loc, agent_pi, agent_scale, agent_y, agent_mask)
        print(f'{loss.shape=}')
        if trajectory_loss_reduction is not None:
            self.trajectory_loss.reduction = trajectory_loss_reduction
        if loss.ndim > 1:  loss = loss.sum(dim=-1)
        assert loss.shape == agent_index.shape, f'{loss.shape=} != {agent_index.shape=}'

        return loss


    def validation_step(self, batch, batch_idx: int) -> None:
        """Validation step for trajectory prediction and OOD detection using trajectory loss"""
        tg.clear_dims()
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
        with torch.set_grad_enabled(False):
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
            grad_agent= loss = loss

        alpha_methods = self.compute_gradient_aggregations(grad_agent)

        # Compute trajectory loss for all agents
        alpha_hat_no_opt = self.true_ood_model.predict_ood_score(local_embed.detach().cpu().numpy())
        alpha_hat_no_opt = torch.from_numpy(alpha_hat_no_opt).float().to(local_embed.device)
        alpha_hat_no_opt_agent = alpha_hat_no_opt[agent_index]


        # OOD detection using trajectory loss as uncertainty score
        is_ood = getattr(data, 'ood', None)
        if is_ood is not None:
            is_ood = is_ood[agent_index]

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