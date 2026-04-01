TRAIN_URL = "https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip"
TEST_URL = "https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_testImages.zip"

.PHONY: help setup get-data train test visualize clean

help:
	@echo "Available commands:"
	@echo "  make setup      - Install required Python dependencies"
	@echo "  make get-data   - Download and extract the hackathon datasets into /data"
	@echo "  make train      - Run the training script (creates segmentation_head.pth)"
	@echo "  make test       - Run the validation script (generates predictions)"
	@echo "  make visualize  - Colorize the raw segmentation masks"
	@echo "  make clean      - Wipe the downloaded data and generated models to reset"

setup:
	@echo "Installing dependencies..."
	pip install -q opencv-python pillow matplotlib tqdm torchvision torch

get-data:
	@echo "Creating data directory and downloading datasets..."
	mkdir -p data
	wget $(TRAIN_URL) -O data/training.zip
	wget $(TEST_URL) -O data/testing.zip
	@echo "Extracting datasets (this might take a moment)..."
	unzip -q data/training.zip -d data/
	unzip -q data/testing.zip -d data/
	@echo "Cleaning up zip files..."
	rm data/training.zip data/testing.zip
	@echo "Data successfully downloaded and extracted into ./data/"

train:
	@echo "Starting training pipeline..."
	python src/train.py

test:
	@echo "Starting testing/inference pipeline..."
	python src/test.py

visualize:
	@echo "Colorizing raw segmentation masks..."
	python src/visualize.py

clean:
	@echo "Cleaning up workspace..."
	rm -rf data/*
	rm -rf predictions/
	rm -rf train_stats/
	rm -f src/segmentation_head.pth
	@echo "Workspace wiped clean."
