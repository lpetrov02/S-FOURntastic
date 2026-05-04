from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig


VOCAB_SIZE = 8192
MAX_LENGTH = 512
MODEL_DIM = 256

lr_options = [3e-4]
difficulty_options = [64]
n_layers = [2]
dataset_size = [100_000]


configs = []
for difficulty in difficulty_options:
    for n in n_layers:
        for lr in lr_options:
            for train_size in dataset_size:
                batch_size = 128 if n <= 4 else 64
                config = TrainConfig(
                    learning_rate=lr,
                    max_epochs=30,
                    weight_decay=0.1,
                    data=DataConfig(
                        train_configs=[
                            MQARConfig(
                                num_examples=train_size,
                                vocab_size=VOCAB_SIZE,
                                input_seq_len=MAX_LENGTH,
                                num_kv_pairs=difficulty,
                            )
                        ],
                        test_configs=[
                            MQARConfig(
                                num_examples=10_000,
                                vocab_size=VOCAB_SIZE,
                                input_seq_len=MAX_LENGTH,
                                num_kv_pairs=difficulty
                            )
                        ],
                        train_batch_size=batch_size,
                        test_batch_size=batch_size,
                    ),
                    model=ModelConfig(
                        vocab_size=VOCAB_SIZE,
                        max_position_embeddings=MAX_LENGTH,
                        sequence_mixer=ModuleConfig(
                            name="zoology.mixers.s4d_base.S4D",
                            kwargs={
                                "dropout": 0.1,
                                "d_state": 16,
                            },
                        ),
                        state_mixer = ModuleConfig(
                            name="zoology.mixers.mlp.MLP", 
                            kwargs={"hidden_mult": 2}
                        ),
                        d_model=MODEL_DIM,
                        block_type="S4DBlock",
                        n_layers=n,
                    ),
                    logger=LoggerConfig(
                        name="tensorboard",
                        project_name=f"S4D_{n}_layers__lr_{lr}__difficulty_{difficulty}",
                    )
                )
                configs.append(config)
