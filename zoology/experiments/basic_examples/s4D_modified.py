import math
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig


VOCAB_SIZE = 8192
configs = []

# (Optional) mimic the paper’s batch-size heuristic for shorter sequences :contentReference[oaicite:2]{index=2}
def bs_for_len(L: int) -> int:
    if L >= 512:
        return 128
    if L >= 256:
        return 256
    return 512

# A denser LR grid helps a lot for Mamba-like models on MQAR :contentReference[oaicite:3]{index=3}
LR_GRID = [
    10**(-4.5), 10**(-4.25), 10**(-4.0),
    10**(-3.75), 10**(-3.5), 10**(-3.25),
    10**(-3.0), 10**(-2.75), 10**(-2.5),
    10**(-2.25), 10**(-2.0),
]


# Seeds used in the paper are (42, 123, 777); keep yours if you prefer :contentReference[oaicite:4]{index=4}
SEEDS = [21] # , 78]  # or [42, 123, 777]

TRAIN_LEN = 256
TEST_LEN  = 256
NUM_KV    = 64

for lr in [6.5e-4, 7.5e-4, 9.5e-4, 2.5e-3, 4.0e-3, 7.0e-3]:
    for d_state in [64]:          # worth sweeping; recurrent models often like width/state :contentReference[oaicite:5]{index=5}
        for d_model in [64]: # paper sweeps width heavily; start small :contentReference[oaicite:6]{index=6}
            for mod in [False]:
                for seed_ in SEEDS:
                    config = TrainConfig(
                        data=DataConfig(
                            train_batch_size=bs_for_len(TRAIN_LEN),
                            test_batch_size=bs_for_len(TRAIN_LEN),  # or bs_for_len(TEST_LEN) if eval is heavy
                            train_configs=[
                                MQARConfig(
                                    num_examples=100_000,
                                    vocab_size=VOCAB_SIZE,
                                    input_seq_len=TRAIN_LEN,
                                    num_kv_pairs=NUM_KV,
                                )
                            ],
                            test_configs=[
                                MQARConfig(
                                    num_examples=3_000,
                                    vocab_size=VOCAB_SIZE,
                                    input_seq_len=TEST_LEN,
                                    num_kv_pairs=NUM_KV,
                                )
                            ],
                            seed=seed_,
                        ),
                        model=ModelConfig(
                            vocab_size=VOCAB_SIZE,
                            d_model=d_model,
                            n_layers=1,

                            # "1-layer without any MLP": make the channel/state mixer an Identity.
                            # (In Zoology-style setups, "1 layer" often means mixer+MLP, but this disables the MLP.)
                            state_mixer=ModuleConfig(
                                name="torch.nn.Identity",
                                kwargs={},
                            ),

                            sequence_mixer=ModuleConfig(
                                name="zoology.mixers.s4d.TokenRoutedS4D",
                                kwargs={
                                    # "use_modification": mod,
                                    "d_state": d_state,
                                    "n_experts": 16,
                                    "top_k": 1,
                                    "dropout": 0.05,
                                    "dt_min": 0.001,
                                    "dt_max": 0.1,
                                    # "d_model": 64,
                                    # "headdim": 8,
                                },
                            ),
                        ),

                        # Paper uses longer training (e.g., 50 epochs) + LR sweeps; you can start smaller :contentReference[oaicite:7]{index=7}
                        max_epochs=30,          # consider 20–50 for final numbers
                        weight_decay=0.001,      # matches common MQAR setups :contentReference[oaicite:8]{index=8}
                        learning_rate=lr,
                        seed=seed_,
                    )
                    configs.append(config)
