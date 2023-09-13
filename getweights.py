import torch

checkpoint = torch.load('weights/latest_model')
torch.save(checkpoint['state_dict'], "./state_dict.pt")