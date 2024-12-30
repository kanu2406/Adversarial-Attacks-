import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from autoattack import AutoAttack
from torchvision.utils import save_image
from torchattacks import PGDL2
from test_project import load_project
from tqdm import tqdm  # For the progress bar


BS = 200

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
ADV_IMAGE_DIR = "./adv_images"
RESULTS_FILE = "./results.csv"
os.makedirs(ADV_IMAGE_DIR, exist_ok=True)

# Load the model
def load_model(model_path, num_classes=10):
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust for num classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load dataset
def load_data(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Evaluate accuracy
def calculate_accuracy(model, inputs, targets):
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
    return correct / inputs.size(0)

# Generate adversarial examples and log results
def generate_and_log_adv_examples(model, dataloader, epsilon=0.03, steps=40, alpha=0.01,max_images=1000):
    # Create AutoAttack for L_inf
    adversary_linf = AutoAttack(model, norm='Linf', eps=epsilon)
    adversary_linf.attacks_to_run = ['apgd-ce']  # Specify attack type for L_inf

    # Create PGD L2 attack
    adversary_l2 = PGDL2(model, eps=epsilon, alpha=alpha, steps=steps)

    results = []
    for idx, (inputs, targets) in enumerate(dataloader):
        if idx >= max_images:
            print("Reached the limit of 1,000 images.")
            break
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Original accuracy
        original_accuracy = calculate_accuracy(model, inputs, targets)

        # Save original image
        orig_save_path = os.path.join(ADV_IMAGE_DIR, f"img_{idx}_original.png")
        save_image(inputs[0], orig_save_path)
        print(f"Saved original image at {orig_save_path}")

        # L_inf attack
        adv_inputs_linf = adversary_linf.run_standard_evaluation(inputs, targets, bs=1)
        linf_save_path = os.path.join(ADV_IMAGE_DIR, f"img_{idx}_atk_Linf.png")
        save_image(adv_inputs_linf[0], linf_save_path)
        linf_accuracy = calculate_accuracy(model, adv_inputs_linf, targets)
        print(f"Saved L_inf adversarial image at {linf_save_path}")

        # PGD L2 attack
        adv_inputs_l2 = adversary_l2(inputs, targets)
        l2_save_path = os.path.join(ADV_IMAGE_DIR, f"img_{idx}_atk_PGD_L2.png")
        save_image(adv_inputs_l2[0], l2_save_path)
        l2_accuracy = calculate_accuracy(model, adv_inputs_l2, targets)
        print(f"Saved PGD L2 adversarial image at {l2_save_path}")

        # Log results
        results.append({
            "Index": idx,
            "Original Image": orig_save_path,
            "L_inf Adv Image": linf_save_path,
            "PGD_L2 Adv Image": l2_save_path,
            "Original Accuracy": original_accuracy,
            "L_inf Accuracy": linf_accuracy,
            "PGD_L2 Accuracy": l2_accuracy
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")

# Evaluate model accuracy after attack
def evaluate_accuracy_after_attack(model, dataloader, epsilon=0.03):
    adversary = AutoAttack(model, norm='Linf', eps=epsilon)
    total_samples = 0
    correct_predictions = 0

    # Use tqdm for progress bar
    for inputs, targets in tqdm(dataloader, desc="Processing images", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate adversarial examples
        adv_inputs = adversary.run_standard_evaluation(inputs, targets, bs=BS)

        # Predict on adversarial examples
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = correct_predictions / total_samples * 100
    print(f"Accuracy after adversarial attack: {accuracy:.2f}%")
    return accuracy

# Main execution
def main():
    # Specify paths and parameters
    project_dir = "/home/jovyan/workspace/assignment3-2024-attack_of_cifar"
    project_module = load_project(project_dir)
    net = project_module.Net()
    net.to(device)
    net.load_for_testing(project_dir=project_dir)
    epsilon = 0.03  # Maximum perturbation
    steps = 40  # Number of steps for PGD
    alpha = 0.01  # Step size for PGD

    # Load model and data
    test_loader = load_data(batch_size=BS)  # Batch size of 1 to process individual images

    # Generate and log adversarial examples
    #generate_and_log_adv_examples(net, test_loader, epsilon=epsilon, steps=steps, alpha=alpha)
    evaluate_accuracy_after_attack(net, test_loader, epsilon=epsilon)

if __name__ == "__main__":
    main()
