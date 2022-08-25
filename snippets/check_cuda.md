
This function first checks CUDA's availability through `torch.cuda.is_available()`. If CUDA is available, it then tries to shift a tensor onto the GPU. If the drivers etc. are working properly, it returns `True`.

```python
import torch

def check_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            a = torch.zeros((2,2))
            a.to(device)
        except e:
            print("CUDA is available, but the drivers do not seem to be installed correctly.")
            print(e)
            return False
        return True
    else:
        return False
```
