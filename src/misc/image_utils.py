import cv2
import numpy as np
import imageio
import torch

@torch.no_grad()
def np2tensor(x: np.array) -> torch.Tensor:
    if x.ndim == 2:
        h, w = x.shape
        x = x.reshape(h, w, 1)
        x = x.repeat(3, axis=-1)

    x = np.transpose(x, (2, 0, 1))
    x = np.ascontiguousarray(x)
    x = torch.from_numpy(x)
    while x.dim() < 4:
        x.unsqueeze_(0)

    x = x.float() / 127.5 - 1

    return x

@torch.no_grad()
def tensor2np(x: torch.Tensor) -> np.array:
    x = 127.5 * (x + 1)
    x = x.round().clamp(min=0, max=255).byte()
    x = x.squeeze(0)
    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = np.ascontiguousarray(x)
    return x

def get_img(img_path: str) -> torch.Tensor:
    x = imageio.imread(img_path)
    x = np2tensor(x)
    return x

def save_img(x: torch.Tensor, img_path: str, scale: float=1) -> None:
    x = tensor2np(x)
    if scale > 1:
        h, w, _ = x.shape
        x = cv2.resize(
            x, (scale * w, scale * h), interpolation=cv2.INTER_NEAREST,
        )

    imageio.imwrite(img_path, x)
    return

@torch.no_grad()
def quantize(x: torch.Tensor) -> torch.Tensor:
    x = 127.5 * (x + 1)
    x = x.clamp(min=0, max=255)
    x = x.round()
    x = x / 127.5 - 1
    return x
