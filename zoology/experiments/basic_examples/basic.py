# from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
# from zoology.data.multiquery_ar import MQARConfig

# VOCAB_SIZE = 8192

# # train_configs = [    
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
# # ]
# # test_configs = [
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=16),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
# #     MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
# # ]


# configs = []

# for lr in [10**(-3.5), 10**(-2.0), 10**(-2.5)]: # , #  #, 7.5e-4, 8e-4, 8.5e-4]: # . 6e-4, 6.5e-4, 7e-4, 
#     for d_state in [64]: # 
#         for mod_dim in [128]: # , , 256
#             for mod in [False, True]: #  
#                 for seed_ in [21,42,63]: #,105]: # ]:
#                     config = TrainConfig(
#                     data=DataConfig(
#                         train_batch_size=256,
#                         test_batch_size=256,
#                         # cache_dir="/path/to/cache/dir"  TODO: add this
#                         train_configs=[
#                             MQARConfig(
#                                 num_examples=20_000,
#                                 vocab_size=VOCAB_SIZE,
#                                 input_seq_len=512,
#                                 num_kv_pairs=64
#                             )
#                         ],
#                         test_configs=[
#                             MQARConfig(
#                                 num_examples=6_000,
#                                 vocab_size=VOCAB_SIZE,
#                                 input_seq_len=612,
#                                 num_kv_pairs=64
#                             )
#                         ]
#                     ),
#                     model=ModelConfig(
#                         vocab_size=VOCAB_SIZE,
#                         d_model=mod_dim,
#                         state_mixer = ModuleConfig(
#                             name="torch.nn.Identity", 
#                             kwargs={"hidden_mult": 2}
#                         ),
#                         sequence_mixer=ModuleConfig(
#                             name="zoology.mixers.mamba.Mamba",
#                             kwargs={
#                                 "use_modification": mod,
#                                 # "dt_min": 0.001,
#                                 "d_state":d_state,
#                                 "headdim":8,
#                                } # , "num_heads": 1}
#                         )
#                     ),
#                     max_epochs=8,
#                     weight_decay=0.1,
#                     learning_rate=lr,# 7.5e-4 if mod is False else lr,
#                     seed=seed_,
#                     )
#                     configs.append(config)
                      
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig


VOCAB_SIZE = 8192
MAX_LENGTH = 1024

config = TrainConfig(
    data=DataConfig(
        # cache_dir="/path/to/cache/dir"  TODO: add this
        train_configs=[
            MQARConfig(
                num_examples=100_000,
                vocab_size=VOCAB_SIZE,
                input_seq_len=MAX_LENGTH,
                num_kv_pairs=64
            )
        ],
        test_configs=[
            MQARConfig(
                num_examples=6_000,
                vocab_size=VOCAB_SIZE,
                input_seq_len=MAX_LENGTH,
                num_kv_pairs=64
            )
        ],
        train_batch_size=32,
        test_betch_size=32,
    ),
    model=ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_LENGTH,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1}
        )
    ),
    
)

configs = [config]