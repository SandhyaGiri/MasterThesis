from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Dataset


def save_model(model: nn.Module, model_params, model_dir):
    torch.save({
        'model_constructor': model.__init__,
        'model_params': model_params,
        'model_state_dict': model.state_dict()
    }, f'{model_dir}/model.tar')

def save_model_with_params_from_ckpt(model: nn.Module, model_dir):
    _, ckpt = load_model(model_dir)
    torch.save({
        'model_constructor': model.__init__,
        'model_params': ckpt['model_params'],
        'model_state_dict': model.state_dict()
    }, f'{model_dir}/model.tar')

def load_model(model_dir, device=None):
    """
    Params
    ------
        model_dir: directory where the model.tar files exists carrying the checkpoint info.
        device: device where the deserialized checkpoint dict should be moved to irrespective of
            the device from where it was saved from. Without this, the checkpoint dict will be 
            desrialized in CPU and move to device where it was saved from, not where it is running
            now.
    """
    ckpt = torch.load(f'{model_dir}/model.tar', map_location=device)
    model_constructor = ckpt['model_constructor']
    model = model_constructor.__self__
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt

def eval_model_on_dataset(model: nn.Module, dataset : Dataset,
                          batch_size: int, device: Optional[torch.device] = None,
                          num_workers=4):
    model.eval()

    testloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader, 0):
            # move tensors to device if available
            if device is not None:
                inputs, labels = map(lambda x: x.to(device),
                                     (inputs, labels))
            # eval the inputs
            logits = model(inputs)

            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    probs = nn.functional.softmax(logits, dim =1)
    labels = torch.cat(labels_list, dim=0)
    return logits.cpu().numpy(), probs.cpu().numpy(), labels.cpu().numpy()

def choose_torch_device(gpus: list):
    """
    Only supports loading tensor on a single GPU or CPU so far.
    """
    if torch.cuda.is_available() and len(gpus) > 0:
        assert torch.cuda.device_count() >= len(gpus)
        device = torch.device("cuda:0")
        print(f"Using device: {torch.cuda.get_device_name(device)} unit {gpus[0]}.")
    else:
        print(f"Using CPU device.")
        device = torch.device("cpu")

    return device