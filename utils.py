import torch


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(path, map_location=map_location))