import torch
import torch.nn as nn

def get_accuracy(y_probs, y_true, device=None, weights=None):
    """
    Calculates accuracy of model's predictsions, given the output probabilities and the truth labels
    """
    if weights is None:
        if device is None:
            accuracy = torch.mean((torch.argmax(y_probs, dim=1) == y_true).to(dtype=torch.float64))
        else:
            accuracy = torch.mean(
                (torch.argmax(y_probs, dim=1) == y_true).to(device, torch.float64))
    else:
        if device is None:
            weights.to(dtype=torch.float64)
            accuracy = torch.mean(
                weights * (torch.argmax(y_probs, dim=1) == y_true).to(dtype=torch.float64))
        else:
            weights.to(device=device, dtype=torch.float64)
            accuracy = torch.mean(
                weights * (torch.argmax(y_probs, dim=1) == y_true).to(device=device,
                                                                      dtype=torch.float64))
    return accuracy

def save_model(model: nn.Module, model_params, model_dir):
    torch.save({
        'model_constructor': model.__init__,
        'model_params': model_params,
        'model_state_dict': model.state_dict()
    }, f'{model_dir}/model.tar')

def load_model(model_dir):
    ckpt = torch.load(f'{model_dir}/model.tar')
    model_constructor = ckpt['model_constructor']
    model = model_constructor.__self__
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt