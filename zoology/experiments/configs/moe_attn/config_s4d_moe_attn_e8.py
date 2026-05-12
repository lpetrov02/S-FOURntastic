from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig


VOCAB_SIZE = 8192
MAX_LENGTH = 512
MODEL_DIM = 256

lr_options = [3e-4]
difficulty_options = [4]
# expert_state_setups = [(8, 2), (8, 16)]
expert_state_setups = [(8, 2)]
n_layers = [2]
dataset_size = [1_000_000]
lbs_options = [None]
attn_dim_options = [64]


configs = []
for difficulty in difficulty_options:
    for n in n_layers:
        for lr in lr_options:
            for train_size in dataset_size:
                for num_experts, state_dim in expert_state_setups:
                    for attn_dim in attn_dim_options:
                        batch_size = 128
                        config = TrainConfig(
                            learning_rate=lr,
                            max_epochs=10,
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
                                    name="zoology.mixers.s4d_moe_attn.S4DMoEAttn",
                                    kwargs={
                                        "dropout": 0.1,
                                        "d_state": state_dim,
                                        "num_experts": num_experts,
                                        "attn_dim": attn_dim,
                                    },
                                ),
                                state_mixer = ModuleConfig(
                                    name="zoology.mixers.mlp.MLP", 
                                    kwargs={"hidden_mult": 2}
                                ),
                                d_model=MODEL_DIM,
                                block_type="S4DMoEAttnBlock",
                                n_layers=n,
                            ),
                            logger=LoggerConfig(
                                name="tensorboard",
                                project_name=f"S4D_attn/Attn_E{num_experts}S{state_dim}_L{n}_dfc{difficulty}_AD{attn_dim}",
                            )
                        )
                        configs.append(config)
