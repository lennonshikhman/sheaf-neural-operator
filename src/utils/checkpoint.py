from __future__ import annotations
from pathlib import Path
import torch

def save_checkpoint(path: str|Path, model, optimizer=None, epoch:int=0, metrics:dict|None=None)->None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state':model.state_dict(), 'optimizer_state': optimizer.state_dict() if optimizer else None, 'epoch':epoch, 'metrics':metrics or {}}, path)
