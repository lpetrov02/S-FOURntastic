from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig


VOCAB_SIZE = 8192
MAX_LENGTH = 512
MODEL_DIM = 256

lr_options = [3e-4]
difficulty_options = [16]
experts_setups = [(32, 2), (32, 4), (32, 1)]
n_layers = [2]
dataset_size = [1_000_000]


configs = []
for difficulty in difficulty_options:
    for n in n_layers:
        for lr in lr_options:
            for train_size in dataset_size:
                for num_experts, top_k in experts_setups:
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
                                name="zoology.mixers.fantastic.FantasticV1",
                                kwargs={
                                    "dropout": 0.1,
                                    "d_state": 16,
                                    "num_experts": num_experts,
                                    "top_k": top_k,
                                },
                            ),
                            state_mixer = ModuleConfig(
                                name="zoology.mixers.mlp.MLP", 
                                kwargs={"hidden_mult": 2}
                            ),
                            d_model=MODEL_DIM,
                            block_type="FantasticV1Block",
                            n_layers=n,
                        ),
                        logger=LoggerConfig(
                            name="tensorboard",
                            project_name=f"fantastic_v1/Fantastic_E{num_experts}A{top_k}_L{n}_dfc{difficulty}_20ep",
                        )
                    )
                    configs.append(config)
