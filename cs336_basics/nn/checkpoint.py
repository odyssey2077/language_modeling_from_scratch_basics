import torch
import os
import typing


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "iteration": iteration}, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model_state"])
    optimizer.load_state_dict(state_dict["optimizer_state"])
    return state_dict["iteration"]