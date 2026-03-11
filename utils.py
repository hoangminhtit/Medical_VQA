import torch


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=map_location)
    
    # Handle architecture changes: remove old gen_head if present
    if isinstance(checkpoint, dict):
        # Filter out gen_head keys if they exist (old architecture)
        filtered_state = {k: v for k, v in checkpoint.items() 
                         if not k.startswith('gen_head.')}
        model.load_state_dict(filtered_state, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)