import torch
import torchvision
import torchvision.transforms as transforms
from autoattack import AutoAttack
from torchattacks import PGDL2
from tqdm import tqdm

import pretrained.pretrained_resnet as pretrained
from model import Net,Base_Net

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Function to collect logits
def collect_logits_periodic_save(clean_model, robust_model, dataloader, epsilon=0.03, alpha=0.01, steps=40, save_path="logits_results.pt", save_interval=1):
    # Initialize attackers
    robust_adversary_linf = AutoAttack(robust_model, norm='Linf', eps=epsilon)
    robust_adversary_l2 = PGDL2(robust_model, eps=epsilon, alpha=alpha, steps=steps)

    clean_adversary_linf = AutoAttack(clean_model, norm='Linf', eps=epsilon)
    clean_adversary_l2 = PGDL2(clean_model, eps=epsilon, alpha=alpha, steps=steps)

    results = []
    batch_counter = 0

    for inputs, labels in tqdm(dataloader, desc="Processing images"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Collect clean logits
        with torch.no_grad():
            clean_logits_clean_model = clean_model(inputs)
            clean_logits_robust_model = robust_model(inputs)

        # Generate adversarial examples (Linf)
        robust_adv_inputs_linf = robust_adversary_linf.run_standard_evaluation(inputs, labels, bs=inputs.size(0))

        # Generate adversarial examples (PGD L2)
        robust_adv_inputs_l2 = robust_adversary_l2(inputs, labels)

        # Generate adversarial examples (Linf)
        clean_adv_inputs_linf = clean_adversary_linf.run_standard_evaluation(inputs, labels, bs=inputs.size(0))

        # Generate adversarial examples (PGD L2)
        clean_adv_inputs_l2 = clean_adversary_l2(inputs, labels)

        # Collect adversarial logits
        with torch.no_grad():
            adv_logits_linf_clean_model = clean_model(clean_adv_inputs_linf)
            adv_logits_linf_robust_model = robust_model(robust_adv_inputs_linf)
            adv_logits_l2_clean_model = clean_model(clean_adv_inputs_l2)
            adv_logits_l2_robust_model = robust_model(robust_adv_inputs_l2)

        # Store results
        for i in range(inputs.size(0)):
            results.append({
                "True Label": labels[i].item(),
                "Clean Logits Clean Model": clean_logits_clean_model[i].cpu(),
                "Clean Logits Robust Model": clean_logits_robust_model[i].cpu(),
                "Adv Linf Logits Clean Model": adv_logits_linf_clean_model[i].cpu(),
                "Adv Linf Logits Robust Model": adv_logits_linf_robust_model[i].cpu(),
                "Adv L2 Logits Clean Model": adv_logits_l2_clean_model[i].cpu(),
                "Adv L2 Logits Robust Model": adv_logits_l2_robust_model[i].cpu(),
            })
            #print(results[-1])

        batch_counter += 1

        # Periodic saving
        if batch_counter % save_interval == 0:
            print(f"Saving intermediate results after {batch_counter} iterations...")
            save_results_to_pt(results, save_path)

    # Final save
    print("Saving final results...")
    save_results_to_pt(results, save_path)

# Save results to .pt file
def save_results_to_pt(results, filename):
    torch.save(results, filename)
    print(f"Results saved to {filename}")

# Main execution
def main():
    # Specify paths and parameters

    epsilon = 0.03  # Maximum perturbation for adversarial attacks
    alpha = 0.01  # Step size for PGD L2
    steps = 40  # Number of steps for PGD L2
    save_path = "logits_results.pt"  # Path to save results
    save_interval = 1  # Save after every 10 batches
    batch_size = 128  # Batch size for data loading

    #clean_model = pretrained.clean_model.eval()
    #robust_model = pretrained.robust_model.eval()

    clean_model = Base_Net().to('cuda')
    robust_model = Base_Net().to('cuda')

    clean_model.load("models/accurate_model.pth")
    robust_model.load("models/robust_model.pth")

    # Load data
    test_loader = load_data(batch_size=batch_size)

    # Collect logits and save results periodically
    collect_logits_periodic_save(clean_model, robust_model, test_loader, epsilon=epsilon, alpha=alpha, steps=steps, save_path=save_path, save_interval=save_interval)

if __name__ == "__main__":
    main()
