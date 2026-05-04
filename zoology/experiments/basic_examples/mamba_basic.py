from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig


VOCAB_SIZE = 8192
MAX_LENGTH = 1024

# lr_options = [1e-4, 3e-4, 1e-3, 1e-2]
# difficulty_options = [4, 16, 64]
# n_layers = [1, 4, 8]

lr_options = [3e-4]
difficulty_options = [4]
n_layers = [8]


configs = []
for difficulty in difficulty_options:
    for n in n_layers:
        for lr in lr_options:
            config = TrainConfig(
                learning_rate=lr,
                max_epochs=10,
                data=DataConfig(
                    train_configs=[
                        MQARConfig(
                            num_examples=20_000,
                            vocab_size=VOCAB_SIZE,
                            input_seq_len=MAX_LENGTH,
                            num_kv_pairs=difficulty,
                        )
                    ],
                    test_configs=[
                        MQARConfig(
                            num_examples=6_000,
                            vocab_size=VOCAB_SIZE,
                            input_seq_len=MAX_LENGTH,
                            num_kv_pairs=difficulty
                        )
                    ],
                    train_batch_size=32,
                    test_batch_size=32,
                ),
                model=ModelConfig(
                    vocab_size=VOCAB_SIZE,
                    max_position_embeddings=MAX_LENGTH,
                    sequence_mixer=ModuleConfig(
                        name="zoology.mixers.mamba.Mamba",
                        kwargs={
                            "dropout": 0.1,
                            "d_state": 16,
                        },
                    ),
                    state_mixer = ModuleConfig(
                        name="zoology.mixers.mlp.MLP", 
                        kwargs={"hidden_mult": 2}
                    ),
                    d_model=512,
                    block_type="MambaBlock",
                    n_layers=n,
                ),
                logger=LoggerConfig(
                    name="tensorboard",
                    project_name=f"Mamba_base_{n}_layers__lr_{lr}__difficulty_{difficulty}",
                )
            )
            configs.append(config)
