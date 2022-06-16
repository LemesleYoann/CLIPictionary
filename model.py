##### Initializes CLIP (from https://github.com/openai/CLIP)
import torch, torchvision, clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

encoder_image = model.encode_image
encoder_text  = model.encode_text
