#!/usr/bin/env python
"""
Train a Latent Gradient Flow Matching model that predicts gradients for latent embeddings.
Uses the LatentGradientsDataset to train on stored optimization gradients.
"""
import os
import json
import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
import logging

from shifts.traj_pred import TrajPredEncoderDecoder
from paths import ROOT


OmegaConf.register_new_resolver('get_model_name', lambda x: x.split('.')[-2])
log = logging.getLogger(__name__)


@hydra.main(config_path=ROOT / "config", config_name="train_flow_matching")
def main(cfg: DictConfig):
    """Main training function"""
    return main_wihout_hydra(cfg)

def main_wihout_hydra(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    encoder_checkpoint_path = (
            ROOT / "experiments/traj_pred/000_enc_dec/model/enc_dec_iros_2023.ckpt"
    )
    encoder_decoder = TrajPredEncoderDecoder.load_from_checkpoint(
        encoder_checkpoint_path,
    )
    # Freeze parameters
    for param in encoder_decoder.parameters():
        param.requires_grad = False
    encoder_decoder.eval()

    # Print configuration
    log.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    # Create the model with velocity model as a subcomponent
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, local_encoder=encoder_decoder.local_encoder, 
                                                        global_interactor=encoder_decoder.global_interactor, decoder=encoder_decoder.decoder)


    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.setup()
    train_dloader = datamodule.train_dataloader()
    val_dloader = datamodule.test_dataloader()

    # Set up logger
    logger = TensorBoardLogger(".", name="flow_matching")

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch:02d}-{" + cfg.tracked_metric + ":.4f}",
        monitor=cfg.tracked_metric,
        auto_insert_metric_name='/' not in cfg.tracked_metric,
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, lr_monitor]

    # Set up trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    if cfg.validate_only:
        assert cfg.checkpoint_path is not None
        model.load_state_dict(torch.load(ROOT / cfg.checkpoint_path)["state_dict"])
        # model.ode_solver = cfg.model.method
        # model.ode_atol = cfg.model.ode_atol
        # model.ode_rtol = cfg.model.ode_rtol
        # print(f"{model.ode_solver=}")
        # print(f"{model.ode_atol=}")
        # print(f"{model.ode_rtol=}")

        return trainer.validate(model, val_dloader)
        return

    # Train model
    trainer.fit(
        model,
        train_dloader,
        val_dloader,
        ckpt_path=ROOT.joinpath(cfg.checkpoint_path) if cfg.checkpoint_path else None,
    )

    log.info(f"Best model path: {checkpoint_callback.best_model_path}")
    log.info(f"Best validation MSE: {checkpoint_callback.best_model_score:.4f}")

    # Run final validation and save results
    log.info("Running final validation round...")
    val_results = trainer.validate(model, val_dloader)
    
    # Prepare results dictionary
    results = {
        "best_model_path": checkpoint_callback.best_model_path,
        "best_model_score": float(checkpoint_callback.best_model_score),
        "final_validation_results": val_results[0] if val_results else {},
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    # Save results to JSON file
    results_path = os.path.join(logger.log_dir, "final_validation_metrics.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Validation results saved to: {results_path}")

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    main()
