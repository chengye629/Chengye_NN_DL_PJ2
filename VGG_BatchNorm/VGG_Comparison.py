import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
# device_id = device_id
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.get_device_name(3))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    print(f"Batch shape: {X.shape}")
    print(f"Labels: {y}")
    ## --------------------
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader):
    ## --------------------
    # Add code as needed
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total
    ## --------------------
    pass

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
# def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
#     model.to(device)
#     learning_curve = [np.nan] * epochs_n
#     train_accuracy_curve = [np.nan] * epochs_n
#     val_accuracy_curve = [np.nan] * epochs_n
#     max_val_accuracy = 0
#     max_val_accuracy_epoch = 0

#     batches_n = len(train_loader)
#     losses_list = []
#     grads = []
#     for epoch in tqdm(range(epochs_n), unit='epoch'):
#         if scheduler is not None:
#             scheduler.step()
#         model.train()

#         loss_list = []  # use this to record the loss value of each step
#         grad = []  # use this to record the loss gradient of each step
#         learning_curve[epoch] = 0  # maintain this to plot the training curve

#         for data in train_loader:
#             x, y = data
#             x = x.to(device)
#             y = y.to(device)
#             optimizer.zero_grad()
#             prediction = model(x)
#             loss = criterion(prediction, y)
#             # You may need to record some variable values here
#             # if you want to get loss gradient, use
#             # grad = model.classifier[4].weight.grad.clone()
#             ## --------------------
#             # Add your code
#             loss.backward()
#             loss_list.append(loss.item())
#             learning_curve[epoch] += loss.item()
#             grad_norm = model.classifier[-1].weight.grad.norm().item() if model.classifier[-1].weight.grad is not None else 0
#             grad.append(grad_norm)
#             optimizer.step()
#             ## --------------------


#             # loss.backward()
#             # optimizer.step()

#         losses_list.append(loss_list)
#         grads.append(grad)
#         display.clear_output(wait=True)
#         f, axes = plt.subplots(1, 2, figsize=(15, 3))

#         learning_curve[epoch] /= batches_n
#         axes[0].plot(learning_curve)

#         # Test your model and save figure here (not required)
#         # remember to use model.eval()
#         ## --------------------
#         # Add code asj8    needed
#         val_acc = get_accuracy(model, val_loader)
#         print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}")
#         axes[1].plot(val_accuracy_curve[:epoch+1]) 
#         ## --------------------

#     return losses_list, grads

def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []
        grad = []
        learning_curve[epoch] = 0

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()

            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()

            grad_norm = model.classifier[-1].weight.grad.norm().item() if model.classifier[-1].weight.grad is not None else 0
            grad.append(grad_norm)

            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        learning_curve[epoch] /= batches_n

        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Avg Loss = {learning_curve[epoch]:.4f}")

    return losses_list, grads, learning_curve, train_accuracy_curve, val_accuracy_curve

def plot_loss_acc_comparison(learning_curve_a, learning_curve_bn,
                             train_accuracy_curve_a, train_accuracy_curve_bn,
                             val_accuracy_curve_a, val_accuracy_curve_bn):
    epochs = range(len(learning_curve_a))
    plt.figure(figsize=(15, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, learning_curve_a, label='VGG_A Loss')
    plt.plot(epochs, learning_curve_bn, label='VGG_BN Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy_curve_a, label='VGG_A Val Acc')
    plt.plot(epochs, val_accuracy_curve_bn, label='VGG_BN Val Acc')
    plt.plot(epochs, train_accuracy_curve_a, '--', label='VGG_A Train Acc')
    plt.plot(epochs, train_accuracy_curve_bn, '--', label='VGG_BN Train Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    # os.makedirs(r"D:/NN_DL/PJ2/codes/VGG_BatchNorm/reports/figures", exist_ok=True)
    project_root = os.getcwd()
    figures_path = os.path.join(project_root, "reports", "figures")
    os.makedirs(figures_path, exist_ok=True)
    plt.savefig(os.path.join(figures_path, "bn_vs_nobn_comparison.png"))
    print("Saved: reports/figures/bn_vs_nobn_comparison.png")


# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(min_curve, max_curve):
    ## --------------------
    # Add your code
    plt.figure(figsize=(10, 5))
    epochs = range(len(min_curve))
    plt.fill_between(epochs, min_curve, max_curve, color='lightblue', alpha=0.5, label='Loss Range')
    plt.plot(min_curve, label='Min Loss')
    plt.plot(max_curve, label='Max Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Landscape over Epochs")
    plt.legend()
    plt.grid(True)
    os.makedirs(r"D:\NN_DL\PJ2\codes\VGG_BatchNorm/reports/figures", exist_ok=True)
    plt.savefig(r"D:\NN_DL\PJ2\codes\VGG_BatchNorm/reports/figures/loss_landscape.png")
    print("Saved: reports/figures/loss_landscape.png")
    ## --------------------
    pass

def plot_loss_landscape_dual(min1, max1, min2, max2, label1='VGG_A', label2='VGG_BatchNorm'):
    plt.figure(figsize=(10, 5))
    epochs = range(len(min1))

    # Model1
    plt.fill_between(epochs, min1, max1, alpha=0.4, label=f'{label1} Loss Range')
    plt.plot(min1, '--', label=f'{label1} Min Loss')
    plt.plot(max1, '--', label=f'{label1} Max Loss')

    # Model2
    plt.fill_between(epochs, min2, max2, alpha=0.4, label=f'{label2} Loss Range')
    plt.plot(min2, '-', label=f'{label2} Min Loss')
    plt.plot(max2, '-', label=f'{label2} Max Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Landscape Comparison")
    plt.legend()
    plt.grid(True)
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/loss_landscape_comparison.png")
    print("Saved: reports/figures/loss_landscape_comparison.png")

def plot_gradient_predictiveness(grads_a, grads_b, label_a='VGG_A', label_b='VGG_BatchNorm'):
    grad_mean_a = [np.mean(g) for g in grads_a]
    grad_std_a = [np.std(g) for g in grads_a]
    grad_mean_b = [np.mean(g) for g in grads_b]
    grad_std_b = [np.std(g) for g in grads_b]

    # Calculate max difference (max - min) within each epoch
    grad_max_diff_a = [np.max(g) - np.min(g) for g in grads_a]
    grad_max_diff_b = [np.max(g) - np.min(g) for g in grads_b]

    epochs = range(len(grad_mean_a))

    plt.figure(figsize=(12, 6))

    # Plot VGG_A's mean gradient and std range
    plt.plot(epochs, grad_mean_a, label=f'{label_a} Mean Grad Norm', color='green', linewidth=1.5)
    plt.fill_between(epochs,
                     np.array(grad_mean_a) - np.array(grad_std_a),
                     np.array(grad_mean_a) + np.array(grad_std_a),
                     alpha=0.2, color='lightgreen', label=f'{label_a} ±1 Std Dev')

    # Plot VGG_BatchNorm's mean gradient and std range
    plt.plot(epochs, grad_mean_b, label=f'{label_b} Mean Grad Norm', color='red', linewidth=1.5)
    plt.fill_between(epochs,
                     np.array(grad_mean_b) - np.array(grad_std_b),
                     np.array(grad_mean_b) + np.array(grad_std_b),
                     alpha=0.2, color='lightcoral', label=f'{label_b} ±1 Std Dev')

    # Plot max difference curves
    plt.plot(epochs, grad_max_diff_a, '--', label=f'{label_a} Max Grad Diff', color='darkgreen', alpha=0.7)
    plt.plot(epochs, grad_max_diff_b, '--', label=f'{label_b} Max Grad Diff', color='darkred', alpha=0.7)

    plt.title('Gradient Predictiveness and Difference Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # os.makedirs(figures_path, exist_ok=True)
    # save_path = os.path.join(figures_path, 'grad_predictiveness_and_max_diff_comparison.png')
    project_root = os.getcwd()
    figures_path = os.path.join(project_root, "reports", "figures")
    os.makedirs(figures_path, exist_ok=True)
    plt.savefig(os.path.join(figures_path, "grad_predictiveness_and_max_diff_comparison.png"))
    print(f"Saved: grad_predictiveness_and_max_diff_comparison.png")
    plt.close()




# Train your model
# feel free to modify
epo = 20
project_root = os.getcwd()

loss_save_path = os.path.join(project_root, "reports")
grad_save_path = os.path.join(project_root, "reports")

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
model_bn = VGG_BatchNorm()
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer_bn = torch.optim.SGD(model_bn.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
criterion_bn = nn.CrossEntropyLoss()
loss, grads, learning_curve_a, train_acc_a, val_acc_a = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
loss_bn, grads_bn, learning_curve_bn, train_acc_bn, val_acc_bn = train(model_bn, optimizer_bn, criterion_bn, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(loss_save_path, 'loss_bn.txt'), loss_bn, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads_bn.txt'), grads_bn, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
# min_curve = []
# max_curve = []
## --------------------
# Add your code
# min_curve = [min(step) for step in loss]
# max_curve = [max(step) for step in loss]
# min_curve_bn = [min(step) for step in loss_bn]
# max_curve_bn = [max(step) for step in loss_bn]
# ## --------------------

# plot_loss_landscape_dual(
#     min_curve, max_curve,           # VGG_A
#     min_curve_bn, max_curve_bn,     # VGG_BatchNorm
#     label1='VGG_A',
#     label2='VGG_BatchNorm'
# )

plot_loss_acc_comparison(
    learning_curve_a, learning_curve_bn,
    train_acc_a, train_acc_bn,
    val_acc_a, val_acc_bn
)

plot_gradient_predictiveness(grads, grads_bn, label_a='VGG_A', label_b='VGG_BatchNorm')

