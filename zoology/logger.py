from pathlib import Path

import trackio as wandb
from torch.nn import Module

from torch.utils.tensorboard import SummaryWriter
from zoology.model import LanguageModel
from zoology.config import LoggerConfig, TrainConfig

from abc import ABC, abstractmethod


class ZoologyLogger(ABC):
    def __init__(self, config: TrainConfig):
        self.no_logger = False

    @abstractmethod
    def log_config(self, config: TrainConfig):
        pass

    @abstractmethod
    def log_model(self, model: LanguageModel, config: TrainConfig):
        pass

    @abstractmethod
    def log(self, metrics: dict):
        pass

    @abstractmethod
    def finish(self):
        pass


class WandbLogger(ZoologyLogger):
    def __init__(self, config: TrainConfig):
        super().__init__(config)
        if config.logger.project_name is None or config.logger.entity is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return

        self.run = wandb.init(
            name=config.run_id,
            entity=config.logger.entity,
            project=config.logger.project_name, 
        )

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        self.run.config.update(config.model_dump(), allow_val_change=True)

    def log_model(
        self, 
        model: LanguageModel,
        config: TrainConfig
    ):
        if self.no_logger:
            return
        
        max_seq_len = max([c.input_seq_len for c in config.data.test_configs])
        wandb.log(
            {
                "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "state_size": model.state_size(sequence_length=max_seq_len),
            }
        )
        wandb.watch(model)

    def log(self, metrics: dict):
        if self.no_logger:
            return
        wandb.log(metrics)
    
    def finish(self):
        if self.no_logger:
            return
        self.run.finish()


class TensorboardLogger(ZoologyLogger):
    def __init__(self, config: TrainConfig):
        super().__init__(config)
        if config.logger.project_name is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return

        log_dir = f"runs/{config.logger.project_name}/{config.run_id}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self._step = 0

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        config_dict = config.model_dump()
        config_text = "\n".join(f"**{k}**: {v}" for k, v in config_dict.items())
        self.writer.add_text("config", config_text, global_step=0)

    def log_model(self, model: LanguageModel, config: TrainConfig):
        if self.no_logger:
            return
        max_seq_len = max([c.input_seq_len for c in config.data.test_configs])
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state_size = model.state_size(sequence_length=max_seq_len)
        self.writer.add_scalar("model/num_parameters", num_parameters, global_step=0)
        self.writer.add_scalar("model/state_size", state_size, global_step=0)

    def log(self, metrics: dict):
        if self.no_logger:
            return
        for key, value in metrics.items():
            if key == "step":
                self._step = value
                continue
            try:
                self.writer.add_scalar(key, value, global_step=self._step)
            except Exception as e:
                print(e)
                pass
        self._step += 1

    def finish(self):
        if self.no_logger:
            return
        self.writer.close()