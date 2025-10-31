# CIFAR Training with Stochastic Weight Averaging (SWA)

A PyTorch training script for CIFAR-10 and CIFAR-100 datasets with optional Stochastic Weight Averaging (SWA) support. This implementation allows you to compare the effects of SWA on model performance across different architectures and optimizers via an easy to use command line script. See _Averaging Weights Leads to Wider Optima and Better Generalization_
by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson for more information about SWA.

## Requirements

```bash
# Core requirements
pip install torch torchvision tabulate tdqm

# Optional: For Weights & Biases logging
pip install wandb
```

## Usage

The general way of using the script follows this format:
```
python3 train.py --dataset=<DATASET> \
                 --data_dir=<PATH> \
                 --results_dir=<DIR> \
                 --model=<MODEL> \
                 --optimizer=<OPTIMIZER>\
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --use_swa \
                 --swa_start=<SWA_START> \
                 --swa_lr=<SWA_LR>
```
See examples below and all arguments below,

### Basic Training (without SWA)

```bash
python train.py --dataset CIFAR10 --model resnet18 --epochs 200
```

### Training with SWA

```bash
python train.py --dataset CIFAR10 --model resnet18 --epochs 200 \
    --use_swa --swa_start 160 --swa_lr 0.05
```

### Training with Weights & Biases Logging
```bash
# First time setup
pip install wandb
wandb login

# Train with wandb logging
python train.py --dataset CIFAR10 --model resnet18 --epochs 200 \
    --use_wandb --use_swa --swa_start 160
```

### Training on CIFAR-100 with Custom Settings

```bash
python train.py --dataset CIFAR100 --model wide_resnet50_2 \
    --optimizer SGD --lr_init 0.1 --wd 5e-4 \
    --epochs 300 --batch_size 256 \
    --use_swa --swa_start 250 --swa_lr 0.05
```

## Command-Line Arguments

### Dataset Parameters
- `--dataset`: Dataset name (`CIFAR10` or `CIFAR100`, default: `CIFAR10`)
- `--data_dir`: Directory for dataset (default: `./data`)
- `--results_dir`: Directory for saving results and checkpoints (default: `./results`)

### Model Parameters
- `--model`: Classification model architecture from [`torchvision.models`](https://docs.pytorch.org/vision/main/models.html#classification) (default: `resnet18`)
  - Examples: `resnet18`, `resnet50`, `vgg16`, `wide_resnet50_2`, `efficientnet_b0`
- `--optimizer`: Optimizer from [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html#algorithms) (default: `SGD`)
  - Examples: `SGD`, `Adam`, `AdamW`, `RMSprop`

### Training Parameters
- `--epochs`: Number of training epochs (default: `200`)
- `--batch_size`: Batch size for training (default: `128`)
- `--lr_init`: Initial learning rate (default: `0.1`)
- `--wd`: Weight decay (default: `5e-4`)
- `--momentum`: Momentum for SGD (default: `0.9`)
- `--eval_freq`: Frequency of test evaluation in epochs (default: `10`)

### SWA Parameters
- `--use_swa`: Enable Stochastic Weight Averaging (flag)
- `--swa_start`: Epoch to start SWA (default: `160`)
- `--swa_lr`: SWA learning rate (default: `0.05`)

### Wandb Parameters
- `--use_wandb`: Use wandb.ai for live logging (flag)
- `--wandb_project`: Wandb project name (default: `'cifar-swa'`)
- `--wandb_entity`: Wandb entity (username or team) (default: `None`)

### Other Parameters
- `--num_workers`: Number of data loading workers (default: `4`)
- `--seed`: Random seed (default: `42`)
- `--save_model`: Saves the model as a `.pth` file (flag)

## Output Files

All output files are saved with unique timestamped filenames in the format:
```
{model}_{dataset}_{epochs}epochs_{SWA}_{timestamp}_{type}
```

### Example Filenames
- Training log: `resnet18_CIFAR10_200epochs_SWA_20241030_143025_training_log.csv`
- Final model: `resnet18_CIFAR10_200epochs_SWA_20241030_143025_final_model.pth`
- SWA model: `resnet18_CIFAR10_200epochs_SWA_20241030_143025_swa_model.pth`

### CSV Log Format
The training log CSV contains the following columns:
- `epoch`: Epoch number (or "SWA_final" for final SWA results)
- `train_loss`: Training loss
- `train_acc`: Training accuracy (%)
- `swa_train_loss`: SWA model training loss (when applicable)
- `swa_train_acc`: SWA model training accuracy (%)
- `test_loss`: Test loss (evaluated at specified frequency)
- `test_acc`: Test accuracy (%)
- `swa_test_loss`: SWA model test loss (when applicable)
- `swa_test_acc`: SWA model test accuracy (%)
- `time`: Time taken for the epoch

## What is Stochastic Weight Averaging (SWA)?

SWA is a simple procedure that improves generalization in deep learning by averaging multiple points along the trajectory of SGD. It typically:
- Improves test accuracy
- Provides better calibrated predictions
- Requires minimal additional computation

The key idea is to average model weights from different epochs in the later stages of training, typically after the learning rate has been reduced.

## Example Comparison

To compare training with and without SWA:

```bash
# Without SWA
python train.py --dataset CIFAR10 --model resnet18 --epochs 200

# With SWA
python train.py --dataset CIFAR10 --model resnet18 --epochs 200 \
    --use_swa --swa_start 160 --swa_lr 0.05
```

## Training Configuration Summary

Before training begins, the script prints a summary:
```
================================================================================
TRAINING CONFIGURATION
================================================================================
Model: resnet18
Dataset: CIFAR10 (10 classes)
Optimizer: SGD
Epochs: 200
Initial LR: 0.1
Weight Decay: 0.0005
Batch Size: 128
SWA: Enabled (starts at epoch 160, LR=0.05)
Results Directory: ./results
Log File: resnet18_CIFAR10_200epochs_SWA_20241030_143025_training_log.csv
================================================================================
```
You can stop at any time by interrupting the terminal using `ctrl + c`.

## References

- [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
- [PyTorch SWA Documentation](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
