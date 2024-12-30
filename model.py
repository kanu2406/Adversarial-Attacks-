#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from mixed_nuts import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32 

'''Basic neural network architecture (from pytorch doc).'''
class Base_Net(nn.Module):

    model_file="models/robust_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)  # 32 * 32
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2) # 16 * 16
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2) # 8 * 8
        self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1  = nn.BatchNorm2d(32)
        self.bn2  = nn.BatchNorm2d(64)
        self.bn3  = nn.BatchNorm2d(64)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
        

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device),weights_only='True'))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))

class Net(nn.Module):
    
    model_file="models/mixed.pth"

    def __init__(self):
        super(Net, self).__init__()
        self.accurate_model = Base_Net().to(device)
        self.robust_model = Base_Net().to(device)
        self.device = device

        # Load pre-trained weights
        #self.accurate_model.load(accurate_model_path)
        #self.robust_model.load(robust_model_path)

        # Parameters for nonlinear transformation and mixing
        self.s = 1.0
        self.p = 1.0
        self.c = 0.0
        self.alpha = 0.5

        # Ensure models are in eval mode
        self.accurate_model.eval()
        self.robust_model.eval()

    def forward(self, x):
        """
        MixedNUTS forward pass.
        Parameters:
            x: Input tensor
        Returns:
            Mixed logits
        """
        accurate_logits = self.accurate_model(x)
        robust_logits = self.robust_model(x)

        # Nonlinear transformation of robust logits
        transformed_robust_logits = nonlinear_transform(robust_logits, self.s, self.p, self.c, clamp_fn=F.gelu)

        # Mix logits
        mixed_logits = mix_logits(accurate_logits, transformed_robust_logits, self.alpha)
        return mixed_logits

    #def optimize_params(self, valid_loader, s_range, p_range, c_range, alpha_range):
    #    """
    #    Optimize the parameters s, p, c, and alpha using validation data.
    #    """
    #    best_params, best_acc = optimize_parameters(valid_loader, s_range, p_range, c_range, alpha_range)
    #    self.s, self.p, self.c, self.alpha = best_params
    #    print("Optimal parameters set:", best_params)

    def save(self, model_file):
        """
        Save the model's parameters and optimal configuration.
        """
        torch.save({
            'accurate_model_state': self.accurate_model.state_dict(),
            'robust_model_state': self.robust_model.state_dict(),
            's': self.s,
            'p': self.p,
            'c': self.c,
            'alpha': self.alpha
        }, model_file)


    def load(self, model_file):
        """
        Load the model's parameters and optimal configuration.
        """
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
        self.accurate_model.load_state_dict(checkpoint['accurate_model_state'])
        self.robust_model.load_state_dict(checkpoint['robust_model_state'])
        self.s = checkpoint['s']
        self.p = checkpoint['p']
        self.c = checkpoint['c']
        self.alpha = checkpoint['alpha']
        self.accurate_model.eval()
        self.robust_model.eval()

    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))


def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

##############################FGSM Attack##################################
def fgsm_attack(model, images, labels, epsilon,targeted =False):
    """
    Implements FGSM attack with a specified epsilon.

    Args:
    - model: The pretrained PyTorch model.
    - images: Original input images.
    - labels: True labels.
    - epsilon: Perturbation magnitude.

    Returns:
    - adversarial_examples: Perturbed images.
    """
    model.eval()
    # Ensure images have gradients enabled
    images.requires_grad = True

    # Forward pass to compute predictions
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    # Generate adversarial perturbations using the sign of the gradient
    perturbations = images.grad.data.sign()

    if not targeted:
      adversarial_examples = images + epsilon * perturbations
    else:
      adversarial_examples = images - epsilon * perturbations

    model.zero_grad()
    # Clip the adversarial examples to stay within valid pixel range [0, 1]
    # adversarial_examples = torch.clamp(adversarial_examples, 0, 1)

    return adversarial_examples

##########################PGD Attack############################
def attack_PGD_flexible(model, x, y, eps, stepsize, iterations, p="inf", targeted=False):
    """
    Implements a flexible PGD attack with support for various norms and targeted/untargeted attacks.

    Args:
    - model: The neural network to attack.
    - x: Input images (original images).
    - y: True labels (untargeted) or target labels (targeted).
    - eps: Radius of the norm-ball for perturbation.
    - stepsize: Step size for gradient update.
    - iterations: Number of PGD steps.
    - p: Norm type ('inf' for l_inf, or any positive integer for l_p norms).
    - targeted: If True, performs a targeted attack; otherwise, untargeted.

    Returns:
    - adv_examples: Adversarial examples (perturbed inputs).
    """
    model.eval()  # Ensure the model is in evaluation mode.
    x.requires_grad = True

    # Initialize perturbations randomly within the norm-ball
    if p == "inf":
        delta = (torch.rand_like(x) * 2 - 1) * eps  # Uniformly distributed in [-eps, eps]
    else:
        delta = torch.randn_like(x)  # Random direction
        norms = torch.norm(delta.view(delta.size(0), -1), dim=1, p=p)
        norms[norms == 0] = 1  # Avoid division by zero
        delta /= norms.view(-1, 1, 1, 1)  # Project onto p-norm ball
        delta *= torch.rand(delta.size(0), 1, 1, 1, device=x.device) * eps

    delta = delta.to(x.device)

    # PGD iterations
    for i in range(iterations):
        adv_examples = x + delta
        adv_examples.requires_grad_(True)

        # Forward pass
        outputs = model(torch.clamp(adv_examples, 0, 1))

        # Compute loss
        loss = F.cross_entropy(outputs, y)
        if targeted:
            loss = -loss  # Minimize loss for targeted attack

        if adv_examples.grad is None:
            adv_examples.retain_grad()  # Allow gradient computation

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Compute gradient step
        if p == "inf":
            gradient = stepsize * adv_examples.grad.sign()
        else:
            gradient = adv_examples.grad
            norms = torch.norm(gradient.view(gradient.size(0), -1), dim=1, p=p)
            norms[norms == 0] = 1  # Avoid division by zero
            gradient /= norms.view(-1, 1, 1, 1)  # Normalize gradient
            gradient *= stepsize

        # Update adversarial examples
        if not targeted:
            delta = delta + gradient
        else:
            delta = delta - gradient

        # Project perturbations back to the norm-ball
        if p == "inf":
            delta = torch.clamp(delta, -eps, eps)
        else:
            norms = torch.norm(delta.view(delta.size(0), -1), dim=1, p=p)
            norms[norms == 0] = 1  # Avoid division by zero
            mask = norms > eps
            delta[mask] /= norms[mask].view(-1, 1, 1, 1)  # Normalize to p-norm ball
            delta[mask] *= eps

    # Create final adversarial examples
    adv_examples = torch.clamp(x + delta, 0, 1)

    return adv_examples





############################Test with FGSM#####################
def test_with_fgsm(model, test_loader, epsilon,targeted = False):
    """
    Tests the model's accuracy under FGSM attack with a single epsilon.

    Args:
    - model: The pretrained model.
    - test_loader: DataLoader for the test dataset.
    - epsilon: Magnitude of perturbation for FGSM.

    Returns:
    - accuracy: Accuracy of the model under FGSM attack.
    """
    correct = 0
    total = 0

    # Set model to evaluation mode
    model.eval()

    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        # Generate adversarial examples
        adv_examples = fgsm_attack(model, images, labels, epsilon,targeted = targeted)

        # Test the model on adversarial examples
        outputs = model(adv_examples)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

############################Test with PGD#####################


def test_with_pgd(model, test_loader, epsilon,stepsize, iterations, p="inf", targeted=False):
    """
    Tests the model's accuracy under FGSM attack with a single epsilon.

    Args:
    - model: The pretrained model.
    - test_loader: DataLoader for the test dataset.
    - epsilon: Magnitude of perturbation for FGSM.

    Returns:
    - accuracy: Accuracy of the model under FGSM attack.
    """
    correct = 0
    total = 0

    # Set model to evaluation mode
    model.eval()

    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        # Generate adversarial examples
        adv_examples = attack_PGD_flexible(model, images, labels, epsilon, stepsize, iterations, p=p, targeted=targeted)

        # Test the model on adversarial examples
        outputs = model(adv_examples)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

########################Adversarial Training###########################


def adversarial_train_model(model, train_loader, pth_filename, num_epochs,
                            epsilon_fgsm=0.03, epsilon_pgd=0.03, alpha=0.004, num_steps=40,
                            clean_weight=0.33, fgsm_weight=0.33, pgd_weight=0.34):
    '''
    Adversarial training of the model using clean, FGSM, and PGD examples.

    Parameters:
    - model: The neural network model to train.
    - train_loader: DataLoader for training data.
    - pth_filename: Filename to save the trained model.
    - num_epochs: Number of epochs for training.
    - epsilon_fgsm: Perturbation magnitude for FGSM.
    - epsilon_pgd: Perturbation magnitude for PGD.
    - alpha: Step size for PGD attack.
    - num_steps: Number of steps for PGD attack.
    - clean_weight: Weight for the loss from clean examples (default 0.33).
    - fgsm_weight: Weight for the loss from FGSM examples (default 0.33).
    - pgd_weight: Weight for the loss from PGD examples (default 0.34).
    '''
    print("Starting adversarial training with clean, FGSM, and PGD examples")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)#optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)

            # Generate adversarial examples using FGSM
            fgsm_inputs = fgsm_attack(model, inputs, labels, epsilon_fgsm,targeted=False)

            # Generate adversarial examples using PGD
            pgd_inputs = attack_PGD_flexible(model, inputs, labels, eps=epsilon_pgd, stepsize=alpha, 
                                             iterations=num_steps, p="inf", targeted=False)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Compute outputs and losses for clean, FGSM, and PGD examples
            clean_outputs = model(inputs)
            clean_loss = criterion(clean_outputs, labels)

            fgsm_outputs = model(fgsm_inputs)
            fgsm_loss = criterion(fgsm_outputs, labels)

            pgd_outputs = model(pgd_inputs)
            pgd_loss = criterion(pgd_outputs, labels)

            # Combined loss
            loss = (clean_weight * clean_loss +
                    fgsm_weight * fgsm_loss +
                    pgd_weight * pgd_loss)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # Print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    # Save the trained model
    torch.save(model.state_dict(), pth_filename)
    print('Model saved in {}'.format(pth_filename))















################################MAIN##################################

def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    parser.add_argument("--epsilon", type=float, default=0.03,
                        help="Perturbation magnitude")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        #train_model(net, train_loader, args.model_file, args.num_epochs)
        adversarial_train_model(net, train_loader, args.model_file, args.num_epochs,
                            epsilon_fgsm=0.03, epsilon_pgd=0.03, alpha=0.004, num_steps=40,
                            clean_weight=0.33, fgsm_weight=0.33, pgd_weight=0.34)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

    acc_fgsm = test_with_fgsm(net, valid_loader, args.epsilon)
    print("Model's accuracy after FGSM attack (valid): {}".format(acc_fgsm))

    acc_pgd = test_with_pgd(net, valid_loader, epsilon=0.02,stepsize=0.004, iterations=40, p=2, targeted=False)
    print("Model's accuracy after PGD attack (valid): {}".format(acc_pgd))


    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()

