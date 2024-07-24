import os
import datetime

from micro_sam.util import _load_checkpoint


def main(name):
	checkpoint_path = os.path.join("models", name, "checkpoints", "organoid_model", "best.pt")
	
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"The trained model path is not found at '{checkpoint_path}'.")

	# Load the state and verify the time taken for getting the best model
	state, _ = _load_checkpoint(checkpoint_path)
	time_in_seconds = state["train_time"]
	
	minutes, seconds = divmod(time_in_seconds, 60)
	hours, minutes = divmod(minutes, 60)

	print("The time taken to achieve the best model -", "%d:%02d:%02d" % (hours, minutes, seconds))


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("name")
	args = parser.parse_args()
	main(args.name)

