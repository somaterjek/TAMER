from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.cli import LightningCLI
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

# For local trial
# LightningCLI(
#     LitTAMER,
#     HMEDatamodule,
#     save_config_kwargs={"overwrite": True},
#     trainer_defaults={"accelerator": "cpu"}

# )

# For distributed training with GPUs and stuff
LightningCLI(
    LitTAMER,
    HMEDatamodule,
    trainer_defaults={
        #"auto_scale_batch_size": "binsearch",
        "strategy": {
            "class_path": "lightning.pytorch.strategies.DDPStrategy",
            "init_args": {"find_unused_parameters": False}
        }
    }
)