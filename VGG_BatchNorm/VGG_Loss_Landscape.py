
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
    # max_val_accuracy = 0
    # max_val_accuracy_epoch = 0

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

    return losses_list

def run_loss_landscape(model_class, model_name, learning_rates, train_loader, val_loader, epochs_n=5):
    models_loss_lists = []

    for lr in learning_rates:
        print(f"\nTraining {model_name} with lr = {lr}")
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss_list = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs_n)

        step_losses = [l for epoch_loss in loss_list for l in epoch_loss]
        models_loss_lists.append(step_losses)
        model_filename = f"{model_name}_lr_{lr}.pth"
        torch.save(model.state_dict(), os.path.join(r"/kaggle/working", model_filename))

        curve_filename = f"{model_name}_lr_{lr}_loss.npy"
        np.savetxt(os.path.join(r"/kaggle/working", curve_filename), loss_list, fmt='%s', delimiter=' ')

    steps_n = min(len(l) for l in models_loss_lists)
    max_curve = []
    min_curve = []
    for i in range(steps_n):
        step_losses = [model_losses[i] for model_losses in models_loss_lists]
        max_curve.append(max(step_losses))
        min_curve.append(min(step_losses))

    return min_curve, max_curve

def plot_dual_loss_landscape(min1, max1, min2, max2,
                             label1='VGG_A',
                             label2='VGG_BatchNorm',
                             color1='green',
                             color2='red'):
    steps = range(len(min1))
    plt.figure(figsize=(10, 6))

    # VGG_A
    plt.fill_between(steps, min1, max1, alpha=0.4, color=color1, label=label1)
    plt.plot(min1, color=color1, linewidth=0.5)
    plt.plot(max1, color=color1, linewidth=0.5)

    # VGG_BatchNorm
    plt.fill_between(steps, min2, max2, alpha=0.4, color=color2, label=label2)
    plt.plot(min2, color=color2, linewidth=0.5)
    plt.plot(max2, color=color2, linewidth=0.5)

    plt.title("Loss Landscape", fontsize=14)
    plt.xlabel("Steps")
    plt.ylabel("Loss Landscape")
    plt.legend()
    plt.grid(True)
    plt.savefig(r"/kaggle/working/loss_landscape_comparison.png")
    print("Saved: reports/figures/loss_landscape_comparison.png")



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



# Train your model
# feel free to modify
epo = 15
# loss_save_path = r"/kaggle/working\reports"
# grad_save_path = r"/kaggle/working\reports"

set_random_seeds(seed_value=2025, device=device)
# np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
# np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')
# np.savetxt(os.path.join(loss_save_path, 'loss_bn.txt'), loss_bn, fmt='%s', delimiter=' ')
# np.savetxt(os.path.join(grad_save_path, 'grads_bn.txt'), grads_bn, fmt='%s', delimiter=' ')

learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]

min_curve_a, max_curve_a = run_loss_landscape(VGG_A, "VGG_A", learning_rates, train_loader, val_loader, epochs_n=epo)
min_curve_bn, max_curve_bn = run_loss_landscape(VGG_BatchNorm, "VGG_BatchNorm", learning_rates, train_loader, val_loader, epochs_n=epo)

plot_dual_loss_landscape(
    min_curve_a, max_curve_a,
    min_curve_bn, max_curve_bn,
    label1='VGG_A',
    label2='VGG_BatchNorm',
)

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
# min_curve = []
# max_curve = []
# ## --------------------
# Add your code
# min_curve = [min(step) for step in loss]
# max_curve = [max(step) for step in loss]
# min_curve_bn = [min(step) for step in loss_bn]
# max_curve_bn = [max(step) for step in loss_bn]
# ## --------------------


