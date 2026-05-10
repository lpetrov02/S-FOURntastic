import argparse
import inspect
import json
import math
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from zoology.config import TrainConfig
from zoology.data.utils import prepare_continuous_data, prepare_data
from zoology.logger import WandbLogger, TensorboardLogger, ZoologyLogger
from zoology.metrics import compute_ce_with_embeddings, compute_mse
from zoology.model import ContinuousInputModel, LanguageModel
from zoology.utils import set_determinism


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        input_type: str = "discrete",
        max_epochs: int = 40,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        loss_type: str = "ce",
        slice_keys: List[str] = [],
        device: Union[str, int] = "cuda",
        logger: ZoologyLogger = None,
        noise_scale_start: float = 1.0,
        noise_scale_end: float = 0.0,
        noise_scale_schedule: str = "cosine",
        mixture_temperature: float = 1.0,
        uniform_topk_eps: float = 0.0,
    ):
        self.model = model  # torch.compile(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.input_type = input_type
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.slice_keys = slice_keys
        self.loss_type = loss_type

        # Routing schedule
        self.noise_scale_start = float(noise_scale_start)
        self.noise_scale_end = float(noise_scale_end)
        self.noise_scale_schedule = noise_scale_schedule
        self.mixture_temperature = float(mixture_temperature)
        self.uniform_topk_eps = float(uniform_topk_eps)

        self.global_step = 0
        self.total_steps = max(1, max_epochs * len(train_dataloader))
        self.current_noise_scale = self.noise_scale_start

        # Detect model.forward kwargs support
        sig = inspect.signature(self.model.forward)
        params = sig.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        self.model_accepts_noise_scale = ("noise_scale" in params) or accepts_kwargs
        self.model_accepts_mixture_temperature = ("mixture_temperature" in params) or accepts_kwargs
        self.model_accepts_uniform_topk_eps = ("uniform_topk_eps" in params) or accepts_kwargs

    def get_noise_scale(self) -> float:
        if self.total_steps <= 1:
            return self.noise_scale_end

        progress = min(1.0, self.global_step / float(self.total_steps - 1))

        if self.noise_scale_schedule == "linear":
            val = self.noise_scale_start + progress * (self.noise_scale_end - self.noise_scale_start)
        elif self.noise_scale_schedule == "cosine":
            val = self.noise_scale_end + 0.5 * (self.noise_scale_start - self.noise_scale_end) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            raise ValueError(
                f"Unknown noise_scale_schedule={self.noise_scale_schedule}. "
                f"Use 'linear' or 'cosine'."
            )

        return val

    def model_forward(self, inputs, **kwargs):
        call_kwargs = dict(kwargs)

        if self.model_accepts_noise_scale:
            call_kwargs["noise_scale"] = self.current_noise_scale
        if self.model_accepts_mixture_temperature:
            call_kwargs["mixture_temperature"] = self.mixture_temperature
        if self.model_accepts_uniform_topk_eps:
            call_kwargs["uniform_topk_eps"] = self.uniform_topk_eps

        return self.model(inputs, **call_kwargs)

    def compute_loss(self, inputs, targets):
        if self.input_type == "continuous":
            all_embeddings = self.model.backbone.embeddings.word_embeddings.weight
            vocab_size = all_embeddings.shape[0]
            embed_dim = all_embeddings.shape[1]
            value_embeddings = all_embeddings[vocab_size // 2:]  # all values as candidates

            outputs = self.model_forward(inputs)
            num_kv_pairs = targets.shape[1]
            outputs = outputs[:, -num_kv_pairs:]

            outputs_flat = outputs.reshape(-1, embed_dim)
            targets_flat = targets.reshape(-1)

            if self.loss_type == "mse":
                target_embeds = value_embeddings[targets_flat]
                loss, _ = compute_mse(outputs_flat, target_embeds)
            else:  # ce or ce_embed
                loss, _ = compute_ce_with_embeddings(
                    outputs_flat, targets_flat, value_embeddings
                )

            logits = outputs_flat @ value_embeddings.T
            preds = logits.argmax(dim=-1).view(targets.shape)
            return loss, preds

        else:  # discrete
            if self.loss_type == "ce":
                hidden = self.model_forward(inputs)  # [B, T, d_model]
                W = self.model.backbone.embeddings.word_embeddings.weight  # [V, d_model]
                logits = hidden @ W.T  # [B, T, V]

                loss = self.loss_fn(
                    rearrange(logits, "b t v -> (b t) v"),
                    targets.flatten(),
                )
                preds = logits.argmax(dim=-1)
                return loss, preds

            elif self.loss_type == "mse":
                embeddings = self.model_forward(inputs, return_embeddings=True)
                target_embeds = self.model.backbone.embeddings.word_embeddings(targets)
                mask = (targets != -100).unsqueeze(-1)
                loss, _ = compute_mse(
                    embeddings[mask.expand_as(embeddings)].view(-1, embeddings.size(-1)),
                    target_embeds[mask.expand_as(target_embeds)].view(-1, target_embeds.size(-1)),
                )
                logits = embeddings @ self.model.backbone.embeddings.word_embeddings.weight.T
                preds = logits.argmax(dim=-1)
                return loss, preds

            elif self.loss_type == "ce_embed":
                embeddings = self.model_forward(inputs, return_embeddings=True)
                value_embeddings = self.model.backbone.embeddings.word_embeddings.weight
                flat_embeds = rearrange(embeddings, "b s d -> (b s) d")
                flat_targets = targets.flatten()
                mask = flat_targets != -100
                loss, _ = compute_ce_with_embeddings(
                    flat_embeds[mask], flat_targets[mask], value_embeddings,
                )
                logits = embeddings @ value_embeddings.T
                preds = logits.argmax(dim=-1)
                return loss, preds

            else:
                raise ValueError(f"Unknown loss_type={self.loss_type}")

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx + 1}/{self.max_epochs}",
        )

        for inputs, targets, slices in iterator:
            self.current_noise_scale = self.get_noise_scale()

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            loss, preds = self.compute_loss(inputs, targets)

            def get_auxiliary_loss(module):
                if hasattr(module, "get_auxiliary_loss"):
                    auxiliary_loss.append(module.get_auxiliary_loss())

            if self.input_type == "discrete":
                auxiliary_loss = []
                self.model.apply(get_auxiliary_loss)
                if auxiliary_loss:
                    loss = loss + sum(auxiliary_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            router_modules = [
                (n, m) for n, m in self.model.named_modules()
                if hasattr(m, "last_entropy")
            ]
            entropy_logs = {}
            if router_modules:
                entropies = [m.last_entropy for _, m in router_modules]
                entropy_logs["train/router_entropy/mean"] = sum(entropies) / len(entropies)
                for name, module in router_modules:
                    entropy_logs[f"train/router_entropy/{name}"] = module.last_entropy

            iterator.set_postfix({
                "loss": loss.item(),
                "noise": round(self.current_noise_scale, 4),
                "tau_mix": round(self.mixture_temperature, 4),
            })
            self.logger.log({
                "train/loss": loss.item(),
                "train/noise_scale": self.current_noise_scale,
                "train/mixture_temperature": self.mixture_temperature,
                "train/uniform_topk_eps": self.uniform_topk_eps,
                "epoch": epoch_idx,
                "step": self.global_step,
                **entropy_logs,
            })

            self.global_step += 1

    def test(self, epoch_idx: int):
        self.model.eval()
        test_loss = 0.0
        results = []

        with torch.no_grad(), tqdm(
            total=len(self.test_dataloader),
            desc=f"Valid Epoch {epoch_idx + 1}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets, slices in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, preds = self.compute_loss(inputs, targets)
                test_loss += loss / len(self.test_dataloader)
                results.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
                iterator.update(1)

            results = pd.DataFrame(results)
            test_accuracy = results["accuracy"].mean()

            metrics = {
                "valid/loss": test_loss.item(),
                "valid/accuracy": test_accuracy.item(),
            }

            for key in self.slice_keys:
                acc_by_slice = results.groupby(key)["accuracy"].mean()
                for value, accuracy in acc_by_slice.items():
                    metrics[f"valid/{key}/accuracy-{value}"] = accuracy

            iterator.set_postfix(metrics)
            self.logger.log({"epoch": epoch_idx, **metrics})

        return metrics

    def fit(self):
        self.model.to(self.device)
        self.current_noise_scale = self.noise_scale_start

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        warmup_epochs = max(1, int(0.1 * self.max_epochs))

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            return 1.0

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )

        final_metrics = None

        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)

            final_metrics = {
                "final_epoch": epoch_idx + 1,
                **metrics,
            }

            if (self.early_stopping_metric is not None) and (
                metrics[self.early_stopping_metric] > self.early_stopping_threshold
            ):
                print(
                    f"Early stopping triggered at epoch {epoch_idx + 1} with "
                    f"{self.early_stopping_metric} = {metrics[self.early_stopping_metric]} "
                    f"> {self.early_stopping_threshold}"
                )
                break

            self.scheduler.step()

        return final_metrics


def save_run_summary(
    config: TrainConfig,
    model: nn.Module,
    final_metrics: dict,
    save_dir: str = "run_summaries",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model.__class__.__name__,
        "backbone_name": getattr(getattr(model, "backbone", None), "__class__", type(None)).__name__,
        "final_metrics": {
            "final_epoch": final_metrics.get("final_epoch"),
            "valid_loss": float(final_metrics.get("valid/loss")),
            "valid_accuracy": float(final_metrics.get("valid/accuracy")),
        },
        "hyperparameters": {
            "seed": getattr(config, "seed", None),
            "input_type": getattr(config, "input_type", None),
            "loss_type": getattr(config, "loss_type", None),
            "max_epochs": getattr(config, "max_epochs", None),
            "learning_rate": getattr(config, "learning_rate", None),
            "weight_decay": getattr(config, "weight_decay", None),
            "noise_scale_start": getattr(config, "noise_scale_start", None),
            "noise_scale_end": getattr(config, "noise_scale_end", None),
            "noise_scale_schedule": getattr(config, "noise_scale_schedule", None),
            "mixture_temperature": getattr(config, "mixture_temperature", None),
            "uniform_topk_eps": getattr(config, "uniform_topk_eps", None),
        },
    }

    model_cfg = getattr(config, "model", None)
    if model_cfg is not None:
        for key in [
            "name",
            "d_model",
            "d_state",
            "n_layers",
            "n_experts",
            "top_k",
            "dropout",
        ]:
            if hasattr(model_cfg, key):
                summary["hyperparameters"][key] = getattr(model_cfg, key)

    save_path = save_dir / f"{run_id}.json"
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    return run_id, save_path


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc,
            }
        )
    return results


def train(config: TrainConfig):
    set_determinism(config.seed)

    logger = None
    if config.logger.name == "wandb":
        logger = WandbLogger(config)
    elif config.logger.name == "tensorboard":
        logger = TensorboardLogger(config)
    else:
        raise ValueError(f"Invalid logger type: {config.logger.name}")
    logger.log_config(config)
    config.print()

    if config.input_type == "continuous":
        model = ContinuousInputModel(config.model)
        train_dataloader, test_dataloader = prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = LanguageModel(config.model)
        train_dataloader, test_dataloader = prepare_data(config.data)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    logger.log_model(model, config=config)

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
        noise_scale_start=getattr(config, "noise_scale_start", getattr(config, "temperature_start", 1.0)),
        noise_scale_end=getattr(config, "noise_scale_end", getattr(config, "temperature_end", 0.0)),
        noise_scale_schedule=getattr(config, "noise_scale_schedule", getattr(config, "temperature_schedule", "cosine")),
        mixture_temperature=getattr(config, "mixture_temperature", 1.0),
        uniform_topk_eps=getattr(config, "uniform_topk_eps", 0.0),
    )

    final_metrics = task.fit()

    run_id, save_path = save_run_summary(
        config=config,
        model=model,
        final_metrics=final_metrics,
        save_dir="run_summaries",
    )

    print(f"Saved run summary to {save_path}")
    logger.log({"run_id": run_id, **final_metrics})
    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train(config)