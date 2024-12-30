import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model import Net,Base_Net

# Helper functions
def layer_normalization(logits):
    """Apply layer normalization to logits."""
    mean = logits.mean(dim=1, keepdim=True)
    std = logits.std(dim=1, keepdim=True)
    return (logits - mean) / (std + 1e-5)

def nonlinear_transform(logits, s, p, c, clamp_fn=F.gelu):
    """
    Nonlinear logit transformation: h_M(s, p, c)
    Parameters:
        logits: Raw logits
        s: Scaling factor
        p: Power exponent
        c: Bias
        clamp_fn: Clamping function (default GELU)
    Returns:
        Transformed logits
    """
    logits = layer_normalization(logits)
    logits = clamp_fn(logits + c)
    logits = s * torch.pow(torch.abs(logits), p) * torch.sign(logits)
    return logits

def mix_logits(accurate_logits, robust_logits, alpha):
    """
    Mix logits from accurate and robust models.
    Parameters:
        accurate_logits: Logits from the accurate model
        robust_logits: Transformed logits from the robust model
        alpha: Mixing weight
    Returns:
        Mixed logits
    """
    accurate_probs = F.softmax(accurate_logits, dim=1)
    robust_probs = F.softmax(robust_logits, dim=1)
    mixed_probs = (1 - alpha) * accurate_probs + alpha * robust_probs
    return torch.log(mixed_probs)

def optimize_parameters(valid_loader, s_range, p_range, c_range, alpha_range, clamp_fn=F.gelu):
    """
    Optimize s, p, c, and alpha using validation data.
    Parameters:
        valid_loader: DataLoader for validation data
        s_range, p_range, c_range, alpha_range: Parameter ranges
        clamp_fn: Clamping function
    Returns:
        Optimal parameters
    """
    best_params = None
    best_accuracy = 0

    for s in s_range:
        for p in p_range:
            for c in c_range:
                for alpha in alpha_range:
                    correct = 0
                    total = 0

                    # Update MixedNUTSNet parameters
                    mixed_nuts.s = s
                    mixed_nuts.p = p
                    mixed_nuts.c = c
                    mixed_nuts.alpha = alpha

                    for inputs, targets in valid_loader:
                        inputs, targets = inputs.to(device), targets.to(device)

                        # Forward pass
                        mixed_logits = mixed_nuts.forward(inputs)
                        predictions = mixed_logits.argmax(dim=1)

                        # Calculate accuracy
                        correct += (predictions == targets).sum().item()
                        total += targets.size(0)

                    accuracy = correct / total
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (s, p, c, alpha)

    return best_params, best_accuracy



if __name__ == "__main__":

    torch.seed()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = Net()
    net.to(device)
    #net.load_for_testing(project_dir=args.project_dir)

    #transform = transforms.Compose([transforms.ToTensor()])
    #cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
    #valid_loader = get_validation_loader(cifar, batch_size=args.batch_size)

    net.accurate_model.load("models/model_clean_acc.pth")
    net.robust_model.load("models/robust_model.pth")

    net.save("models/mixed.pth")



    #acc_nat = test_natural(net, valid_loader, num_samples = args.num_samples)
    #print("Model nat accuracy (test): {}".format(acc_nat))


