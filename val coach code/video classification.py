from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.io import read_video
import torch

model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

