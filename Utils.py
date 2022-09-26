import torch
import pynvml
import os
import numpy as np
from torch_cka import CKA

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement()*param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
        

def get_best_gpu():
    """Return gpu (:class:`torch.device`) with largest free memory."""
    assert torch.cuda.is_available()
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys() is not None:
        cuda_devices = [
            int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        ]
    else:
        cuda_devices = range(deviceCount)

    assert max(cuda_devices) < deviceCount
    deviceMemory = []
    for i in cuda_devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d" % (best_device_index))


def accuracy(pred, target):
    """Computes the accuracy of the model."""
    correct = pred.eq(target).sum().item()
    total = len(pred)
    return correct / total


def compute_layerwise_diff(params1, params2):
    """Compute the difference between two sets of parameters."""
    assert params1.keys() == params2.keys()
    diff_table = dict.fromkeys(params1.keys())
    with torch.no_grad():
        for param_name in params1.keys():
            diff = torch.norm(params1[param_name].cpu(
            ).float() - params2[param_name].cpu().float())
            diff_table[param_name] = diff.detach().numpy().item()
        return diff_table

def compute_cka_similarity(model1, model2, loader, device, layer2compare=None):
    with torch.no_grad():
        cka = CKA(model1, model2, 
                  model1_name='global',
                  model2_name='local',
                  model1_layers=layer2compare,
                  model2_layers=layer2compare,
                  device=device)
        cka.compare(loader, loader)
        result = cka.export()
        return result  
    


