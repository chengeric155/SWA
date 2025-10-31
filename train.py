import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from tabulate import tabulate
import csv

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR10/CIFAR100 with optional SWA')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='CIFAR10', 
                        choices=['CIFAR10', 'CIFAR100'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for dataset')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for saving results and checkpoints')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model architecture from torchvision.models')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Optimizer from torch.optim')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Image size (auto-detected for ViT models, default 32 for CNNs)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr_init', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (for SGD)')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Frequency of test evaluation')
    
    # SWA parameters
    parser.add_argument('--use_swa', action='store_true',
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa_start', type=int, default=160,
                        help='Epoch to start SWA')
    parser.add_argument('--swa_lr', type=float, default=0.05,
                        help='SWA learning rate')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='cifar-swa',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team)')
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(args):
    """Load CIFAR10 or CIFAR100 dataset"""
    # Determine image size - auto-detect for ViT or use specified/default
    if args.image_size is not None:
        img_size = args.image_size
    else:
        # Auto-detect: ViT needs 224x224, CNNs use 32x32
        img_size = 224 if 'vit' in args.model.lower() else 32
    
    # Data augmentation for training
    if img_size != 32:
        # For larger images (e.g., ViT): resize and adjust padding
        padding = int(img_size * 0.125)  # Proportional padding (4/32 = 0.125)
        transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        # For 32x32 images (standard CIFAR size)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    if args.dataset == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, test_loader, num_classes


def get_model(args, num_classes):
    """Initialize model from torchvision.models"""
    try:
        model_fn = getattr(torchvision.models, args.model)
        model = model_fn(num_classes=num_classes)
    except AttributeError:
        raise ValueError(f"Model {args.model} not found in torchvision.models")
    
    return model


def get_optimizer(args, model):
    """Initialize optimizer from torch.optim"""
    try:
        optimizer_class = getattr(optim, args.optimizer)
    except AttributeError:
        raise ValueError(f"Optimizer {args.optimizer} not found in torch.optim")
    
    # Create optimizer with appropriate parameters
    if args.optimizer == 'SGD':
        optimizer = optimizer_class(
            model.parameters(), lr=args.lr_init, 
            momentum=args.momentum, weight_decay=args.wd)
    else:
        optimizer = optimizer_class(
            model.parameters(), lr=args.lr_init, weight_decay=args.wd)
    
    return optimizer


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / total, 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, 100. * correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Initialize wandb if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Install with: pip install wandb")
            print("Continuing without wandb logging...")
            args.use_wandb = False
        else:
            # Generate run name with optimizer
            swa_suffix = '_SWA' if args.use_swa else ''
            run_name = f'{args.model}_{args.dataset}_{args.optimizer}_{args.epochs}epochs{swa_suffix}'
            
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args)
            )
    
    # Setup directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    print(f'Loading {args.dataset}...')
    train_loader, test_loader, num_classes = get_dataset(args)
    
    # Initialize model
    print(f'Initializing {args.model}...')
    model = get_model(args, num_classes).to(device)
    
    # Initialize optimizer
    optimizer = get_optimizer(args, model)
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize SWA if requested
    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
        print(f'SWA will start at epoch {args.swa_start}')
    
    criterion = nn.CrossEntropyLoss()
    
    # Generate unique filenames with timestamp and optimizer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    swa_suffix = '_SWA' if args.use_swa else ''
    base_filename = f'{args.model}_{args.dataset}_{args.optimizer}_{args.epochs}epochs{swa_suffix}_{timestamp}'
    log_file = os.path.join(args.results_dir, f'{base_filename}_training_log.csv')
    
    # Print training summary
    print('\n' + '='*80)
    print('TRAINING CONFIGURATION')
    print('='*80)
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset} ({num_classes} classes)')
    print(f'Optimizer: {args.optimizer}')
    print(f'Epochs: {args.epochs}')
    print(f'Initial LR: {args.lr_init}')
    print(f'Weight Decay: {args.wd}')
    print(f'Batch Size: {args.batch_size}')
    # Detect image size used
    img_size = args.image_size if args.image_size is not None else (224 if 'vit' in args.model.lower() else 32)
    print(f'Image Size: {img_size}x{img_size}')
    if args.use_swa:
        print(f'SWA: Enabled (starts at epoch {args.swa_start}, LR={args.swa_lr})')
    else:
        print(f'SWA: Disabled')
    print(f'Results Directory: {args.results_dir}')
    print(f'Log File: {base_filename}_training_log.csv')
    print('='*80 + '\n')
    
    # Training loop
    results = []
    
    print('\nStarting training...\n')
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Update learning rate
        if args.use_swa and epoch > args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Prepare result row
        result_row = {
            'epoch': epoch,
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
        }
        
        # Evaluate SWA model if applicable
        if args.use_swa and epoch >= args.swa_start:
            swa_loss, swa_acc = evaluate(swa_model, train_loader, criterion, device)
            result_row['swa_train_loss'] = f'{swa_loss:.4f}'
            result_row['swa_train_acc'] = f'{swa_acc:.2f}%'
        else:
            result_row['swa_train_loss'] = '-'
            result_row['swa_train_acc'] = '-'
        
        # Evaluate on test set
        if epoch == 1 or epoch == args.epochs or epoch % args.eval_freq == 0:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            result_row['test_loss'] = f'{test_loss:.4f}'
            result_row['test_acc'] = f'{test_acc:.2f}%'
            
            # Evaluate SWA on test set
            if args.use_swa and epoch >= args.swa_start:
                swa_test_loss, swa_test_acc = evaluate(swa_model, test_loader, criterion, device)
                result_row['swa_test_loss'] = f'{swa_test_loss:.4f}'
                result_row['swa_test_acc'] = f'{swa_test_acc:.2f}%'
            else:
                result_row['swa_test_loss'] = '-'
                result_row['swa_test_acc'] = '-'
        else:
            result_row['test_loss'] = '-'
            result_row['test_acc'] = '-'
            result_row['swa_test_loss'] = '-'
            result_row['swa_test_acc'] = '-'
        
        result_row['time'] = f'{epoch_time:.2f}s'
        results.append(result_row)
        
        # Log to wandb if enabled
        if args.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'lr': optimizer.param_groups[0]['lr'],
                'time/epoch': epoch_time
            }
            
            if args.use_swa and epoch >= args.swa_start:
                log_dict['swa_train/loss'] = swa_loss
                log_dict['swa_train/acc'] = swa_acc
            
            if epoch == 1 or epoch == args.epochs or epoch % args.eval_freq == 0:
                log_dict['test/loss'] = test_loss
                log_dict['test/acc'] = test_acc
                
                if args.use_swa and epoch >= args.swa_start:
                    log_dict['swa_test/loss'] = swa_test_loss
                    log_dict['swa_test/acc'] = swa_test_acc
            
            wandb.log(log_dict, step=epoch)
        
        # Print table - only show header every 40 epochs
        table = tabulate([result_row], headers='keys', tablefmt='simple', floatfmt='8.4f')
        if epoch % 40 == 1:  # Show header on first epoch and every 40 epochs
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
        
        # Save to CSV file
        with open(log_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    # Update batch normalization statistics for SWA model
    if args.use_swa:
        print('\nUpdating SWA batch normalization statistics...')
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        # Final SWA evaluation
        swa_test_loss, swa_test_acc = evaluate(swa_model, test_loader, criterion, device)
        print(f'Final SWA Test Loss: {swa_test_loss:.4f}, Test Acc: {swa_test_acc:.2f}%')
        
        # Log final SWA results to wandb
        if args.use_wandb:
            wandb.log({
                'swa_final/test_loss': swa_test_loss,
                'swa_final/test_acc': swa_test_acc
            })
        
        # Append final SWA results to CSV
        final_swa_row = {
            'epoch': 'SWA_final',
            'train_loss': '-',
            'train_acc': '-',
            'swa_train_loss': '-',
            'swa_train_acc': '-',
            'test_loss': '-',
            'test_acc': '-',
            'swa_test_loss': f'{swa_test_loss:.4f}',
            'swa_test_acc': f'{swa_test_acc:.2f}%',
            'time': '-'
        }
        results.append(final_swa_row)
        
        # Update CSV with final SWA results
        with open(log_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    # Save final models
    if args.save_model:
        print('\nSaving models...')
        model_file = os.path.join(args.results_dir, f'{base_filename}_final_model.pth')
        torch.save(model.state_dict(), model_file)
        print(f'Saved model to: {model_file}')
        
        if args.use_swa:
            swa_model_file = os.path.join(args.results_dir, f'{base_filename}_swa_model.pth')
            torch.save(swa_model.state_dict(), swa_model_file)
            print(f'Saved SWA model to: {swa_model_file}')
            
        # Log model artifacts to wandb
        if args.use_wandb:
            wandb.save(model_file)
            if args.use_swa:
                wandb.save(swa_model_file)
    else:
        print('\nSkipping model saving (--save_model not specified)')
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
    
    print(f'\nTraining complete! Results saved to {args.results_dir}')


if __name__ == '__main__':
    main()