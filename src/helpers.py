import torch 

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the time indices corresponding to a `record_every` parameter.
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )


if __name__=="__main__":
    pass
