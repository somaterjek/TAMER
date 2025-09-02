# Minimal script to train TAMER on ds_small (no CLI)
import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from tamer.lit_tamer import LitTAMER
from tamer.datamodule import HMEDatamodule

def main():
	# Set up the datamodule for ds_small
	datamodule = HMEDatamodule(
		folder="data/ds_small",
		test_folder="validation",  # or "test" for test set
		max_size=32e4,
		scale_to_limit=True,
		train_batch_size=2,
		eval_batch_size=2,
		num_workers=0,
		scale_aug=False,
	)

	# Set up the model with minimal working hyperparameters
	model = LitTAMER(
		d_model=128,
		growth_rate=16,
		num_layers=4,
		nhead=4,
		num_decoder_layers=2,
		dim_feedforward=256,
		dropout=0.1,
		dc=32,
		cross_coverage=False,
		self_coverage=False,
		beam_size=1,
		max_len=150,
		alpha=0.6,
		early_stopping=False,
		temperature=1.0,
		learning_rate=1.0,
		patience=10,
		milestones=[10, 20],
		vocab_size=86,
	)


	# Set up logger
	logger = TensorBoardLogger("lightning_logs", name=None)

	# Set up checkpoint callback to save in the versioned log dir
	checkpoint_callback = ModelCheckpoint(
		dirpath=os.path.join(logger.log_dir, "checkpoints"),
		save_top_k=1,
		monitor="val_loss",
		mode="min",
		save_last=True
	)

	trainer = L.Trainer(
		max_epochs=5,
		accelerator="auto",
		devices=1,
		log_every_n_steps=1,
		logger=logger,
		callbacks=[checkpoint_callback],
	)
	trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
	main()
