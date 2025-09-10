import os
import json
import numpy as np
import torch
import torch.nn as nn
import argparse
import shutil
import pandas as pd
import time
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from itt.configuration_itt import IttConfig
from itt.modeling_itt_cross import IttCrossSequenceClassifier
from itt.utils import (
    calculate_metrics,
    calculate_classification_metrics,
    IndividualFeatureDataset,
    collate_fn,
    AverageMeter,
    LinearWarmupScheduler,
)

def parse_args():
    parser = argparse.ArgumentParser(description='IT-Transformer Cross-Attention Training')
    
    # Data related arguments
    parser.add_argument('--features-folder', type=str, required=True,
                      help='path to folder containing pre-calculated feature .npz files (one per sample)')
    parser.add_argument('--label-csv', type=str, default=None,
                      help='path to CSV file containing file IDs and labels (for random splitting)')
    
    # Pre-split dataset arguments
    parser.add_argument('--train-csv', type=str, default=None,
                      help='path to pre-split training CSV file (first column: IDs, second column: labels, no header)')
    parser.add_argument('--val-csv', type=str, default=None,
                      help='path to pre-split validation CSV file (first column: IDs, second column: labels, no header)')
    parser.add_argument('--test-csv', type=str, default=None,
                      help='path to pre-split test CSV file (first column: IDs, second column: labels, no header)')
    
    # Dataset splitting arguments (for random splitting)
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='ratio of data to use for training (only used with --label-csv)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                      help='ratio of data to use for validation (only used with --label-csv)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                      help='ratio of data to use for testing (only used with --label-csv)')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='random seed for dataset splitting (only used with --label-csv)')
    
    parser.add_argument('--save-path', type=str, default=None,
                      help='path to save model checkpoints (if None, will not save)')
    
    # Task related arguments
    parser.add_argument('--task', choices=['regression', 'classification'],
                      default='regression', help='task type')
    parser.add_argument('--num-labels', type=int, default=1,
                      help='number of labels for classification')
    
    # Cross-attention specific arguments
    parser.add_argument('--use-cross-attention-mask', action='store_true', default=True,
                      help='whether to use the special cross-attention mask (default: True)')
    
    # Training related arguments
    parser.add_argument('--batch-size', type=int, default=32,
                      help='batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                      help='batch size for validation')
    parser.add_argument('--epochs', type=int, default=5,
                      help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='initial learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                      help='number of epochs for learning rate warmup (default: min(1, 0.1 * total_epochs))')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                      help='weight decay')
    
    # Model related arguments
    parser.add_argument('--n0', type=int, default=1,
                      help='number of nodes in first layer')
    parser.add_argument('--d0', type=int, default=750,
                      help='dimension of first layer')
    parser.add_argument('--n1', type=int, default=7,
                      help='number of nodes in second layer')
    parser.add_argument('--d1', type=int, default=750,
                      help='dimension of second layer')
    parser.add_argument('--n2', type=int, default=42,
                      help='number of nodes in third layer')
    parser.add_argument('--d2', type=int, default=500,
                      help='dimension of third layer')
    parser.add_argument('--n-conv', type=int, default=4,
                      help='layer number for graph part')
    parser.add_argument('--classifier-head-method', choices=['cls', 'mean', 'split_mean'],
                      default='cls', help='method for regression/classification: cls (use [CLS] token), mean (average all tokens), split_mean (average each part separately)')
    
    # Other arguments
    parser.add_argument('--disable-cuda', action='store_true',
                      help='disable CUDA')
    parser.add_argument('--disable-mps', action='store_true',
                      help='disable MPS (Metal Performance Shaders) for Apple Silicon')
    parser.add_argument('--print-freq', type=int, default=10,
                      help='print frequency')
    
    # Add standardization argument
    parser.add_argument('--standardize-labels', action='store_true',
                      help='whether to standardize labels (fits scaler on training data only, applies to all splits)')
    parser.add_argument('--standardize-features', action='store_true',
                      help='whether to standardize features (fits scalers on training data only, applies to all splits)')
    
    # Add feature scaler argument
    parser.add_argument('--feature-scaler', type=str, default=None,
                      help='path for feature scaler ')
    # Add pretrained model argument
    parser.add_argument('--pretrained-model', type=str, default=None,
                      help='Path to a pretrained model checkpoint (from pretraining). If provided, will load weights and add a new regression/classification head for finetuning.')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to a checkpoint file to resume training from')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers for data loading (default: 2)')
    parser.add_argument('--save-top-n', type=int, default=1,
                      help='Number of top models to save based on validation performance (default: 1)')
    parser.add_argument('--early-stop-patience', type=int, default=None,
                      help='Number of epochs to wait for improvement before early stopping (default: None, no early stopping)')
    parser.add_argument('--disable-weighted-loss', action='store_true',
                      help='Disable weighted loss for classification (use standard CrossEntropyLoss)')
    return parser.parse_args()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # Save the model configuration as a dictionary
    model_config = state['model'].config if 'model' in state else state['config']
    config_dict = model_config.to_dict() if hasattr(model_config, 'to_dict') else model_config.__dict__
    state['config'] = config_dict
    # Remove model object if present
    if 'model' in state:
        del state['model']
    # Save using torch.save
    torch.save(state, filename)
    if is_best:
        best_model_path = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
        shutil.copyfile(filename, best_model_path)
    return None

def format_time(seconds):
    """Format seconds into a human-readable string."""
    return str(timedelta(seconds=int(seconds)))

class EarlyStopping:
    """
    Early stopping handler to stop training when validation metric doesn't improve.
    """
    def __init__(self, patience, task='regression'):
        self.patience = patience
        self.task = task
        self.counter = 0
        self.best_metric = float('inf') if task == 'regression' else float('-inf')
        self.early_stop = False
        self.is_better = lambda x, y: x < y if task == 'regression' else x > y
    
    def __call__(self, current_metric):
        """
        Check if training should stop early.
        
        Args:
            current_metric: Current validation metric value
            
        Returns:
            bool: True if training should stop early, False otherwise
        """
        if self.patience is None:
            return False
        
        if self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def get_best_metric(self):
        """Get the best metric value seen so far."""
        return self.best_metric
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.early_stop = False

class TopNModelManager:
    """
    Manages the top N models based on validation performance.
    """
    def __init__(self, n, save_path, task='regression'):
        self.n = n
        self.save_path = save_path
        self.task = task
        self.models = []  # List of (metric, epoch, state_dict) tuples
        self.metric_key = 'orig_mae' if task == 'regression' else 'accuracy'
        self.is_better = lambda x, y: x < y if task == 'regression' else x > y
    
    def update(self, metric, epoch, state_dict):
        """
        Update the top N models list.
        
        Args:
            metric: Validation metric value
            epoch: Current epoch number
            state_dict: Model state dictionary
        """
        # Add new model
        self.models.append((metric, epoch, state_dict))
        
        # Sort by metric (ascending for regression, descending for classification)
        self.models.sort(key=lambda x: x[0], reverse=(self.task == 'classification'))
        
        # Keep only top N
        if len(self.models) > self.n:
            self.models = self.models[:self.n]
    
    def _save_models(self):
        """Save all top N models to disk."""
        for i, (metric, epoch, state_dict) in enumerate(self.models):
            model_path = os.path.join(self.save_path, f'model_top_{i+1}_epoch_{epoch}.pth.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'metric': metric,
                'rank': i + 1
            }, model_path)
    
    def get_best_model_state(self):
        """Get the state dict of the best model."""
        if self.models:
            return self.models[0][2]  # Return state_dict of best model
        return None
    
    def get_all_model_states(self):
        """Get state dicts of all top N models."""
        return [model[2] for model in self.models]
    
    def print_summary(self):
        """Print summary of top N models."""
        print(f"\nTop {len(self.models)} models:")
        for i, (metric, epoch, state_dict) in enumerate(self.models):
            print(f"  Rank {i+1}: Epoch {epoch}, {self.metric_key} = {metric:.5e}")
    
    def save_final_models(self):
        """Save the final top N models to disk (called only at the end of training)."""
        if not self.models:
            print("No models to save")
            return
        
        print(f"Saving final top {len(self.models)} models to disk...")
        for i, (metric, epoch, state_dict) in enumerate(self.models):
            model_path = os.path.join(self.save_path, f'model_top_{i+1}_epoch_{epoch}.pth.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'metric': metric,
                'rank': i + 1
            }, model_path)
            print(f"  Saved: {os.path.basename(model_path)} (metric: {metric:.5e})")
        print("✓ All top N models saved to disk")


def train(train_loader, model, criterion, optimizer, scheduler, device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, labels = batch
        # Handle labels differently for regression vs classification
        if args.task == 'regression':
            labels = labels.to(device).view(-1, 1)
        else:  # classification
            labels = labels.to(device).view(-1).long()  # 1D tensor for CrossEntropyLoss
        part0, part1, part2, graph_data_tuple = inputs
        part0 = part0.to(dtype=torch.float32).to(device)
        part1 = part1.to(dtype=torch.float32).to(device)
        part2 = part2.to(dtype=torch.float32).to(device)
        atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx = graph_data_tuple
        atom_fea = atom_fea.to(dtype=torch.float32).to(device)
        nbr_fea = nbr_fea.to(dtype=torch.float32).to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        cluster_indices = cluster_indices.to(device)
        crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
        graph_data = (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)
        optimizer.zero_grad()
        outputs = model(input_part0=part0, input_part1=part1, input_part2=part2, graph_data=graph_data, use_cross_attention_mask=args.use_cross_attention_mask)
        predictions = outputs[0]
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Batch-based scheduler update
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), labels.size(0))
        if i % args.print_freq == 0:
            print(f'Loss scaled: {losses.val:.4f} ({losses.avg:.4f})')
    return losses.avg

def validate(val_loader, model, device, epoch, args, label_scaler):
    model.eval()
    all_preds = []
    all_labels = []
    all_orig_labels = []
    val_losses = AverageMeter()
    
    # Use unweighted loss for validation (fair evaluation)
    val_criterion = nn.MSELoss() if args.task == 'regression' else nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            # Store original labels if using scaling (only for regression)
            if label_scaler is not None and args.task == 'regression':
                all_orig_labels.append(label_scaler.inverse_transform(labels.numpy().reshape(-1, 1)).flatten())
            
            # Handle labels differently for regression vs classification
            if args.task == 'regression':
                labels = labels.to(device).view(-1, 1)
            else:  # classification
                labels = labels.to(device).view(-1).long()  # 1D tensor for CrossEntropyLoss
            
            part0, part1, part2, graph_data_tuple = inputs
            # Convert input tensors to float32 before moving to device
            part0 = part0.to(device)
            part1 = part1.to(device)
            part2 = part2.to(device)
            
            atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx = graph_data_tuple
            # Convert graph features to float32 before moving to device
            atom_fea = atom_fea.to(dtype=torch.float32).to(device)
            nbr_fea = nbr_fea.to(dtype=torch.float32).to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            cluster_indices = cluster_indices.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
            graph_data = (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)
            
            # Get model outputs without passing labels
            outputs = model(input_part0=part0, input_part1=part1, input_part2=part2, 
                          graph_data=graph_data, use_cross_attention_mask=args.use_cross_attention_mask)
            predictions = outputs[0]
            
            loss = val_criterion(predictions, labels)
            val_losses.update(loss.item(), labels.size(0))
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Use appropriate metrics function based on task
    if args.task == 'regression':
        metrics = calculate_metrics(all_labels, all_preds)
    else:  # classification
        metrics = calculate_classification_metrics(all_labels, all_preds)
    metrics['val_loss'] = val_losses.avg
    
    if label_scaler is not None:
        all_orig_labels = np.concatenate(all_orig_labels, axis=0)
        orig_preds = label_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
        orig_metrics = calculate_metrics(all_orig_labels, orig_preds)
        for key, value in orig_metrics.items():
            metrics[f'orig_{key}'] = value
    
    print(f"Validation Metrics: Epoch {epoch}")
    print("Metrics:", end=" ")
    for key, value in metrics.items():
        if not key.startswith('orig_') and key != 'val_loss':
            if args.task == 'regression' and key in ['mae', 'mse', 'rmse']:
                print(f"{key}: {value:.4e}", end=" ")
            else:
                print(f"{key}: {value:.4f}", end=" ")
    print()
    if label_scaler is not None and args.task == 'regression':
        print("Original space metrics:", end=" ")
        for key, value in metrics.items():
            if key.startswith('orig_'):
                if key[5:] in ['mae', 'mse', 'rmse']:
                    print(f"{key[5:]}: {value:.4e}", end=" ")
                else:
                    print(f"{key[5:]}: {value:.4f}", end=" ")
        print()
    
    return metrics

def evaluate_and_save_predictions(model, test_loader, test_split_data, device, args, label_scaler):
    """
    Evaluate the model on the test set and save predictions.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        test_split_data: Tuple of (indices, file_ids, labels) for test split
        device: Device to run evaluation on
        args: Command line arguments
        label_scaler: Label scaler if labels were standardized
    """
    print("\nEvaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            
            # Handle labels differently for regression vs classification
            if args.task == 'regression':
                labels = labels.to(device).view(-1, 1)
            else:  # classification
                labels = labels.to(device).view(-1).long()  # 1D tensor for CrossEntropyLoss
            
            part0, part1, part2, graph_data_tuple = inputs
            # Convert input tensors to float32 before moving to device
            part0 = part0.to(device)
            part1 = part1.to(device)
            part2 = part2.to(device)
            
            atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx = graph_data_tuple
            # Convert graph features to float32 before moving to device
            atom_fea = atom_fea.to(dtype=torch.float32).to(device)
            nbr_fea = nbr_fea.to(dtype=torch.float32).to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            cluster_indices = cluster_indices.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
            graph_data = (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)
            
            # Get model outputs
            outputs = model(input_part0=part0, input_part1=part1, input_part2=part2, 
                          graph_data=graph_data, use_cross_attention_mask=args.use_cross_attention_mask)
            predictions = outputs[0]
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Get test split data
    test_indices, test_file_ids, test_labels_scaled = test_split_data
    
    # Calculate metrics in scaled space
    if args.task == 'regression':
        metrics = calculate_metrics(all_labels, all_preds)
    else:  # classification
        metrics = calculate_classification_metrics(all_labels, all_preds)
    
    # If using scaling, calculate metrics in original space
    if label_scaler is not None and args.task == 'regression':
        # Use the entire test_labels_scaled from test_split_data and inverse transform it
        all_orig_labels = label_scaler.inverse_transform(test_labels_scaled.reshape(-1, 1)).flatten()
        # Transform predictions back to original scale
        orig_preds = label_scaler.inverse_transform(all_preds)
        orig_metrics = calculate_metrics(all_orig_labels, orig_preds)
        # Add original scale metrics with 'orig_' prefix
        for key, value in orig_metrics.items():
            metrics[f'orig_{key}'] = value
    
    print("\nTest Set Metrics:")
    print("Metrics:", end=" ")
    for key, value in metrics.items():
        if not key.startswith('orig_'):
            if args.task == 'regression' and key in ['mae', 'mse', 'rmse']:
                print(f"{key}: {value:.4e}", end=" ")
            else:
                print(f"{key}: {value:.4f}", end=" ")
    print()
    if label_scaler is not None and args.task == 'regression':
        print("Original space metrics:", end=" ")
        for key, value in metrics.items():
            if key.startswith('orig_'):
                if key[5:] in ['mae', 'mse', 'rmse']:
                    print(f"{key[5:]}: {value:.4e}", end=" ")
                else:
                    print(f"{key[5:]}: {value:.4f}", end=" ")
        print()
    
    # Save predictions alongside test labels
    if args.task == 'classification':
        # For classification: use mapped labels (consistent with model predictions)
        predictions_df = pd.DataFrame({
            0: test_file_ids,  # file_id
            1: all_labels.flatten(),  # mapped labels (consistent with predictions)
            2: all_preds.flatten()  # predictions
        })
    else:
        # For regression: handle both scaled and original labels
        if label_scaler is not None:
            # When scaling is used: test_labels_scaled contains scaled labels
            # Use the entire test_labels_scaled from test_split_data and inverse transform it
            all_orig_labels = label_scaler.inverse_transform(test_labels_scaled.reshape(-1, 1)).flatten()
            predictions_df = pd.DataFrame({
                0: test_file_ids,  # file_id
                1: all_orig_labels,  # original labels
                2: test_labels_scaled,  # scaled labels
                3: all_preds.flatten(),  # predictions in scaled space
                4: label_scaler.inverse_transform(all_preds).flatten()  # predictions in original space
            })
        else:
            # When no scaling: test_labels_scaled contains original labels
            predictions_df = pd.DataFrame({
                0: test_file_ids,  # file_id
                1: test_labels_scaled,  # original labels (same as scaled when no scaling)
                2: all_preds.flatten()  # predictions
            })
    
    # Create predictions file path
    predictions_path = os.path.join(args.save_path if args.save_path else '.', 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False, header=False)
    print(f"\nPredictions saved to: {predictions_path}")
    
    return metrics

def evaluate_and_save_predictions_ensemble(model, model_states, test_loader, test_split_data, device, args, label_scaler):
    """
    Evaluate ensemble of models on the test set and save predictions.
    
    Args:
        model: The model template (will be loaded with different states)
        model_states: List of model state dictionaries
        test_loader: DataLoader for test data
        test_split_data: Tuple of (indices, file_ids, labels) for test split
        device: Device to run evaluation on
        args: Command line arguments
        label_scaler: Label scaler if labels were standardized
    """
    print(f"\nEvaluating ensemble of {len(model_states)} models on test set...")
    model.eval()
    all_ensemble_preds = []
    all_labels = []
    
    # Store individual model predictions
    individual_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            
            # Handle labels differently for regression vs classification
            if args.task == 'regression':
                labels = labels.to(device).view(-1, 1)
            else:  # classification
                labels = labels.to(device).view(-1).long()  # 1D tensor for CrossEntropyLoss
            
            part0, part1, part2, graph_data_tuple = inputs
            # Convert input tensors to float32 before moving to device
            part0 = part0.to(device)
            part1 = part1.to(device)
            part2 = part2.to(device)
            
            atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx = graph_data_tuple
            # Convert graph features to float32 before moving to device
            atom_fea = atom_fea.to(dtype=torch.float32).to(device)
            nbr_fea = nbr_fea.to(dtype=torch.float32).to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            cluster_indices = cluster_indices.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
            graph_data = (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)
            
            # Get predictions from all models
            batch_preds = []
            for i, state_dict in enumerate(model_states):
                model.load_state_dict(state_dict)
                outputs = model(input_part0=part0, input_part1=part1, input_part2=part2, 
                              graph_data=graph_data, use_cross_attention_mask=args.use_cross_attention_mask)
                predictions = outputs[0]
                batch_preds.append(predictions.cpu().numpy())
            
            # Average predictions from all models
            batch_preds = np.array(batch_preds)  # [num_models, batch_size, num_classes] for classification, [num_models, batch_size, 1] for regression
            ensemble_pred = np.mean(batch_preds, axis=0)  # [batch_size, num_classes] and [batch_size, 1] for regression
            
            all_ensemble_preds.append(ensemble_pred)
            all_labels.append(labels.cpu().numpy())
            individual_preds.append(batch_preds.transpose(1, 0, 2))  # [batch_size, num_models, ...]
    
    all_ensemble_preds = np.concatenate(all_ensemble_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    individual_preds = np.concatenate(individual_preds, axis=0)  # [total_samples, num_models, ...]
    individual_preds = individual_preds.transpose(1, 0, 2)  # [num_models, total_samples, ...]
    
    # Get test split data
    test_indices, test_file_ids, test_labels_scaled = test_split_data
    
    # Calculate metrics in scaled space
    if args.task == 'regression':
        metrics = calculate_metrics(all_labels, all_ensemble_preds)
    else:  # classification
        metrics = calculate_classification_metrics(all_labels, all_ensemble_preds)
    
    # If using scaling, calculate metrics in original space
    if label_scaler is not None and args.task == 'regression':
        # Use the entire test_labels_scaled from test_split_data and inverse transform it
        all_orig_labels = label_scaler.inverse_transform(test_labels_scaled.reshape(-1, 1)).flatten()
        # Transform ensemble predictions back to original scale
        orig_ensemble_preds = label_scaler.inverse_transform(all_ensemble_preds)
        orig_metrics = calculate_metrics(all_orig_labels, orig_ensemble_preds)
        # Add original scale metrics with 'orig_' prefix
        for key, value in orig_metrics.items():
            metrics[f'orig_{key}'] = value
    
    print("\nEnsemble Test Set Metrics:")
    print("Metrics:", end=" ")
    for key, value in metrics.items():
        if not key.startswith('orig_'):
            if args.task == 'regression' and key in ['mae', 'mse', 'rmse']:
                print(f"{key}: {value:.4e}", end=" ")
            else:
                print(f"{key}: {value:.4f}", end=" ")
    print()
    if label_scaler is not None and args.task == 'regression':
        print("Original space metrics:", end=" ")
        for key, value in metrics.items():
            if key.startswith('orig_'):
                if key[5:] in ['mae', 'mse', 'rmse']:
                    print(f"{key[5:]}: {value:.4e}", end=" ")
                else:
                    print(f"{key[5:]}: {value:.4f}", end=" ")
        print()
    
    # Save ensemble predictions alongside test labels
    if args.task == 'classification':
        # For classification: use mapped labels (consistent with model predictions)
        # Convert ensemble probabilities to predicted class labels
        ensemble_predicted_classes = np.argmax(all_ensemble_preds, axis=1)  # [total_samples]
        
        predictions_df = pd.DataFrame({
            0: test_file_ids,  # file_id
            1: all_labels.flatten(),  # mapped labels (consistent with predictions)
            2: ensemble_predicted_classes  # ensemble predicted class labels
        })
        
        # Add individual model predictions
        for i in range(len(model_states)):
            col_idx = 3 + i
            # individual_preds[i] has shape [total_samples, num_classes] for classification
            # We need to get the predicted class (argmax) for each sample
            predicted_classes = np.argmax(individual_preds[i], axis=1)  # [total_samples]
            predictions_df[col_idx] = predicted_classes
    else:
        # For regression: handle both scaled and original labels
        if label_scaler is not None:
            # When scaling is used: test_labels_scaled contains scaled labels
            # Use the entire test_labels_scaled from test_split_data and inverse transform it
            all_orig_labels = label_scaler.inverse_transform(test_labels_scaled.reshape(-1, 1)).flatten()
            predictions_df = pd.DataFrame({
                0: test_file_ids,  # file_id
                1: all_orig_labels,  # original labels
                2: test_labels_scaled,  # scaled labels
                3: all_ensemble_preds.flatten(),  # ensemble predictions in scaled space
                4: label_scaler.inverse_transform(all_ensemble_preds).flatten()  # ensemble predictions in original space
            })
            
            # Add individual model predictions (both scaled and original)
            for i in range(len(model_states)):
                col_idx_scaled = 5 + i  # scaled predictions
                col_idx_orig = 5 + len(model_states) + i  # original predictions
                predictions_df[col_idx_scaled] = individual_preds[i].flatten()
                # Transform individual predictions to original scale
                individual_orig_preds = label_scaler.inverse_transform(individual_preds[i])
                predictions_df[col_idx_orig] = individual_orig_preds.flatten()
        else:
            # When no scaling: test_labels_scaled contains original labels
            predictions_df = pd.DataFrame({
                0: test_file_ids,  # file_id
                1: test_labels_scaled,  # original labels (same as scaled when no scaling)
                2: all_ensemble_preds.flatten()  # ensemble predictions
            })
            
            # Add individual model predictions
            for i in range(len(model_states)):
                col_idx = 3 + i
                predictions_df[col_idx] = individual_preds[i].flatten()
    
    # Create predictions file path
    predictions_path = os.path.join(args.save_path if args.save_path else '.', 'test_predictions_ensemble.csv')
    predictions_df.to_csv(predictions_path, index=False, header=False)
    print(f"\nEnsemble predictions saved to: {predictions_path}")
    
    return metrics

def main():
    args = parse_args()
    
    # Set device with MPS support for Apple Silicon
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not args.disable_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Determine data loading method
    use_presplit = (args.train_csv is not None and args.val_csv is not None and args.test_csv is not None)
    use_random_split = (args.label_csv is not None)
    
    if use_presplit and use_random_split:
        raise ValueError("Cannot use both --label-csv (random split) and pre-split CSV files (--train-csv, --val-csv, --test-csv) at the same time")
    elif not use_presplit and not use_random_split:
        raise ValueError("Must provide either --label-csv (for random split) or all three pre-split CSV files (--train-csv, --val-csv, --test-csv)")
    
    if use_presplit:
        # Load pre-split CSV files
        print("Using pre-split CSV files...")
        print(f"Loading training labels from: {args.train_csv}")
        print(f"Loading validation labels from: {args.val_csv}")
        print(f"Loading test labels from: {args.test_csv}")
        
        # Load each split
        train_df = pd.read_csv(args.train_csv, header=None)
        val_df = pd.read_csv(args.val_csv, header=None)
        test_df = pd.read_csv(args.test_csv, header=None)
        
        train_ids = train_df[0].values
        train_labels = train_df[1].values
        val_ids = val_df[0].values
        val_labels = val_df[1].values
        test_ids = test_df[0].values
        test_labels = test_df[1].values
        
        # Combine all labels for label mapping (for classification)
        all_labels = np.concatenate([train_labels, val_labels, test_labels])
        all_ids = np.concatenate([train_ids, val_ids, test_ids])
        
        print(f"Pre-split dataset sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
        
    else:
        # Load single CSV file for random splitting
        print(f"Loading labels from: {args.label_csv}")
        df = pd.read_csv(args.label_csv, header=None)
        all_ids = df[0].values  # First column contains file IDs
        all_labels = df[1].values     # Second column contains labels
        
        print(f"Total samples: {len(all_ids)}")
        
        # Split the dataset
        print(f"Splitting dataset with ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
            
        # Set random seed for reproducibility
        np.random.seed(args.random_seed)
        
        # Calculate split sizes
        n_samples = len(all_ids)
        n_train = int(n_samples * args.train_ratio)
        n_val = int(n_samples * args.val_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        
        # Create split data
        train_ids = all_ids[train_indices]
        train_labels = all_labels[train_indices]
        val_ids = all_ids[val_indices]
        val_labels = all_labels[val_indices]
        test_ids = all_ids[test_indices]
        test_labels = all_labels[test_indices]
    
    # Handle non-standard labels for classification
    if args.task == 'classification':
        unique_labels = np.unique(all_labels)
        print(f"Found unique labels: {unique_labels}")
        
        # Check if number of unique labels matches num_labels
        if len(unique_labels) != args.num_labels:
            print(f"ERROR: Expected {args.num_labels} unique labels but found {len(unique_labels)}")
            print("Available labels:", unique_labels)
            raise ValueError(f"Number of unique labels ({len(unique_labels)}) does not match num_labels ({args.num_labels})")
        
        # Map all unique labels to standard indices [0, 1, 2, ...]
        print(f"Mapping {len(unique_labels)} unique labels to standard indices [0, 1, 2, ...]")
        # Sort unique labels to ensure consistent mapping
        sorted_unique_labels = np.sort(unique_labels)
        label_mapping = {old: new for new, old in enumerate(sorted_unique_labels)}
        
        # Apply mapping to all splits
        train_labels = np.array([label_mapping[label] for label in train_labels])
        val_labels = np.array([label_mapping[label] for label in val_labels])
        test_labels = np.array([label_mapping[label] for label in test_labels])
        
        print(f"✓ Label mapping: {label_mapping}")
        print(f"Final label distribution: {np.bincount(np.concatenate([train_labels, val_labels, test_labels]))}")
        print(f"Final unique labels: {np.unique(np.concatenate([train_labels, val_labels, test_labels]))}")
    
    # Verify that all label IDs have corresponding feature files
    all_split_ids = np.concatenate([train_ids, val_ids, test_ids])
    print(f"Verifying feature files exist for all {len(all_split_ids)} samples...")
    missing_files = []
    for file_id in all_split_ids:
        feature_file = os.path.join(args.features_folder, f"{file_id}.npz")
        if not os.path.exists(feature_file):
            missing_files.append(file_id)
    
    if missing_files:
        raise ValueError(f"Missing feature files for IDs: {missing_files[:10]}... (showing first 10)")
    
    print("✓ All feature files found")
    
    # Create split data structure (same format as before)
    splits = {
        'train': (np.arange(len(train_ids)), train_ids, train_labels),
        'val': (np.arange(len(val_ids)), val_ids, val_labels),
        'test': (np.arange(len(test_ids)), test_ids, test_labels)
    }
    
    # Apply standardization if requested
    label_scaler = None  # Initial
    
    # Save label mapping info for classification tasks
    label_mapping_info = None
    if args.task == 'classification':
        # Create mapping info for reference
        unique_labels = np.unique(all_labels)
        sorted_unique_labels = np.sort(unique_labels)
        label_mapping = {old: new for new, old in enumerate(sorted_unique_labels)}
        
        label_mapping_info = {
            'task': 'classification',
            'num_classes': args.num_labels,
            'original_labels': sorted_unique_labels.tolist(),
            'mapped_labels': list(range(args.num_labels)),
            'mapping': {k: v for k, v in label_mapping.items()}
        }
    
    if args.standardize_labels and args.task == 'regression':
        print("Standardizing labels using training data only...")
        train_labels = splits['train'][2]
        val_labels = splits['val'][2]
        test_labels = splits['test'][2]
        
        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(train_labels.reshape(-1, 1))
        # scaler.fit(labels.reshape(-1, 1))
        label_scaler = scaler
        
        # Transform all splits
        train_labels_scaled = scaler.transform(train_labels.reshape(-1, 1)).flatten()
        val_labels_scaled = scaler.transform(val_labels.reshape(-1, 1)).flatten()
        test_labels_scaled = scaler.transform(test_labels.reshape(-1, 1)).flatten()
        
        # Update splits with scaled labels
        splits['train'] = (splits['train'][0], splits['train'][1], train_labels_scaled)
        splits['val'] = (splits['val'][0], splits['val'][1], val_labels_scaled)
        splits['test'] = (splits['test'][0], splits['test'][1], test_labels_scaled)
        
        print("✓ Labels standardized")
    
    # Set default values for graph module parameters
    args.orig_atom_fea_len = 92  # From atom_init.json
    args.nbr_fea_len = 41  # From GaussianDistance with radius=8.0, step=0.2

    # Create model configuration
    if args.pretrained_model is not None:
        print(f"Loading pretrained model from: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device, weights_only=False)
        # Load config from checkpoint (dict)
        if isinstance(checkpoint['config'], dict):
            config_dict = checkpoint['config']
        else:
            config_dict = IttConfig.to_dict(checkpoint['config'])
        config = IttConfig(**config_dict)
        # Set regression/classification method and num_labels for finetuning
        config.classifier_head_method = args.classifier_head_method
        config.num_labels = args.num_labels
        model = IttCrossSequenceClassifier(config)
        # Load pretrained weights (ignore head weights)
        state_dict = checkpoint['state_dict']
        # Remove head weights (regression/classification head)
        head_keys = [k for k in state_dict.keys() if 'classifier' in k or 'regression_head' in k or 'classification_head' in k]
        for k in head_keys:
            del state_dict[k]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights. Missing keys: {missing}, Unexpected keys: {unexpected}")
        model.to(device)
        print(model)
        # Print number of trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")
    else:
        config = IttConfig(
            n0=args.n0, d0=args.d0,
            n1=args.n1, d1=args.d1,
            n2=args.n2, d2=args.d2,
            classifier_head_method=args.classifier_head_method,
            num_labels=args.num_labels,
            orig_atom_fea_len=args.orig_atom_fea_len,
            nbr_fea_len=args.nbr_fea_len,
            n_conv=args.n_conv,
        )
        # Initialize model
        model = IttCrossSequenceClassifier(config)
        model.to(device)
        print(model)

    # Feature standardization: require and load scaler if requested
    feature_scaler = None
    if args.standardize_features:
        if not args.feature_scaler:
            raise ValueError("--feature-scaler must be provided if --standardize-features is set. Use the utility in main/itt/utils.py to create one.")
        if not os.path.exists(args.feature_scaler):
            raise FileNotFoundError(f"Feature scaler file not found: {args.feature_scaler}")
        print(f"Loading feature scaler from: {args.feature_scaler}")
        feature_scaler = load(args.feature_scaler)
        print("✓ Feature scalers loaded")

    # Prepare data loaders
    print("Creating data loaders...")
    
    # Create datasets
    train_dataset = IndividualFeatureDataset(splits['train'], args.features_folder, feature_scaler)
    val_dataset = IndividualFeatureDataset(splits['val'], args.features_folder, feature_scaler)
    test_dataset = IndividualFeatureDataset(splits['test'], args.features_folder, feature_scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Initialize optimizer
    base_lr = args.lr
    # Use a single optimizer for all parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    # Calculate warmup steps
    if args.warmup_epochs is None:
        args.warmup_epochs = min(1, int(0.05 * args.epochs))
        print(f"Using automatic warmup epochs: {args.warmup_epochs}")
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = LinearWarmupScheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load optimizer states (handle both old and new format)
        if 'optimizer' in checkpoint:
            # Old format with single optimizer - load into optimizer_other
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer state from checkpoint")
        
        # Load scheduler states (handle both old and new format)
        if 'scheduler' in checkpoint:
            # Old format with single scheduler - load into scheduler_other
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Loaded scheduler state from checkpoint")
        
        # Get starting epoch
        start_epoch = checkpoint['epoch']
        
        # Load feature scaler if present
        if 'feature_scaler' in checkpoint and checkpoint['feature_scaler'] is not None:
            feature_scaler = checkpoint['feature_scaler']
            print("Loaded feature scaler from checkpoint")
        
        print(f"Resuming training from epoch {start_epoch}")

    # Initialize timing variables
    total_start_time = time.time()
    best_metric = float('inf') if args.task == 'regression' else float('-inf')
    
    # Initialize loss criterion
    if args.task == 'regression':
        criterion = nn.MSELoss()
    else:  # classification
        if args.disable_weighted_loss:
            print("Using standard CrossEntropyLoss (weighted loss disabled)")
            criterion = nn.CrossEntropyLoss()
        else:
            # Calculate class weights for balanced training
            train_labels_for_weights = splits['train'][2]  # Get training labels
            class_counts = np.bincount(train_labels_for_weights)
            total_samples = len(train_labels_for_weights)
            num_classes = len(class_counts)
            class_weights = total_samples / (num_classes * class_counts)
            class_weights_tensor = torch.FloatTensor(class_weights).to(device)
            
            print(f"Class distribution in training set: {class_counts}")
            print(f"Class weights for balanced training: {class_weights}")
            
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    best_model_state = None
    
    # Initialize top N model manager
    top_n_manager = TopNModelManager(args.save_top_n, args.save_path, args.task) if args.save_path else None
    
    # Initialize early stopping handler
    early_stopping = EarlyStopping(args.early_stop_patience, args.task) if args.early_stop_patience else None

    print("\nStarting training...")
    print(f"Total epochs: {args.epochs}")
    print(f"Total steps per epoch: {len(train_loader)}")
    print(f"Total training steps: {len(train_loader) * args.epochs}")
    if early_stopping:
        print(f"Early stopping enabled with patience: {args.early_stop_patience}")
    else:
        print("Early stopping disabled")
    args.use_cross_attention_mask = False
    print(f"Using cross-attention mask: {args.use_cross_attention_mask}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, scheduler, device, args)

        # Calculate timing information
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        avg_epoch_time = total_time / (epoch + 1)
        estimated_remaining_time = avg_epoch_time * (args.epochs - epoch - 1)
        
        # Print timing information at the end of epoch
        print(f"Time: {format_time(epoch_time)} | Total: {format_time(total_time)} | ETA: {format_time(estimated_remaining_time)}")
        
        # Evaluate on validation set
        val_metrics = validate(val_loader, model, device, epoch, args, label_scaler)

        # Report current learning rate after each epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6e}")
        
        # scheduler.step() # This line is removed as per the new_code, as the scheduler is now batch-based.
        
        # Save checkpoint if needed
        if args.save_path is not None:
            # Select appropriate metric for best model selection
            if args.task == 'regression':
                # For regression: use MAE (lower is better)
                metric_key = 'orig_mae' if label_scaler is not None else 'mae'
                current_metric = val_metrics[metric_key]
                is_best = current_metric < best_metric
                best_metric = min(current_metric, best_metric)
            else:
                # For classification: use accuracy (higher is better)
                metric_key = 'accuracy'
                current_metric = val_metrics[metric_key]
                is_best = current_metric > best_metric
                best_metric = max(current_metric, best_metric)
            
            # Update top N models
            if top_n_manager is not None:
                top_n_manager.update(current_metric, epoch + 1, model.state_dict())
            
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric': best_metric,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': model.config,
                'feature_scaler': feature_scaler,
                'label_scaler': label_scaler,
                'label_mapping_info': label_mapping_info,
                'args': vars(args) if hasattr(args, '__dict__') else args
            }
            save_checkpoint(checkpoint_state, is_best, os.path.join(args.save_path, 'checkpoint.pth.tar'))
            if is_best:
                best_model_path = os.path.join(args.save_path, 'model_best.pth.tar')
                print(f"Model saved at epoch {epoch+1}, saved to {best_model_path}")
                best_model_state = model.state_dict()  # Save best weights in memory
        
        # Check early stopping
        if early_stopping:
            if early_stopping(current_metric):
                print(f"\nEarly stopping triggered! No improvement for {args.early_stop_patience} epochs.")
                print(f"Best {metric_key}: {early_stopping.get_best_metric():.5e}")
                print(f"Stopping training at epoch {epoch + 1}")
                break
    
    # Print final timing summary
    total_training_time = time.time() - total_start_time
    actual_epochs = epoch + 1 if 'epoch' in locals() else args.epochs
    print(f"\nTraining completed! Total time: {format_time(total_training_time)}")
    print(f"Actual epochs trained: {actual_epochs}/{args.epochs}")
    
    if early_stopping and early_stopping.early_stop:
        print(f"Training stopped early due to no improvement for {args.early_stop_patience} epochs")
    
    # Print label mapping information for classification
    if args.task == 'classification' and label_mapping_info is not None:
        print("\n" + "="*50)
        print("LABEL MAPPING INFORMATION")
        print("="*50)
        print(f"Classification task with {label_mapping_info['num_classes']} classes")
        print(f"Original labels: {label_mapping_info['original_labels']}")
        print(f"Mapped to: {label_mapping_info['mapped_labels']}")
        print(f"Mapping: {label_mapping_info['mapping']}")
        print("Note: Labels are sorted and mapped to sequential indices [0, 1, 2, ...]")
        print("="*50)

    # Load models for testing
    if top_n_manager is not None and len(top_n_manager.models) > 0:
        print("Using ensemble of top N models for testing...")
        top_n_manager.print_summary()
        top_n_manager.save_final_models()
        
        # Get all model states
        model_states = top_n_manager.get_all_model_states()
        
        # Evaluate ensemble
        test_metrics = evaluate_and_save_predictions_ensemble(
            model, model_states, test_loader, splits['test'], device, args, label_scaler
        )
    elif best_model_state is not None:
        print("Loading best model for testing...")
        model.load_state_dict(best_model_state)
        print("✓ Best model loaded for testing")
        
        # Evaluate on test set and save predictions
        test_metrics = evaluate_and_save_predictions(model, test_loader, splits['test'], device, args, label_scaler)
    else:
        print("No models available for testing!")
        return

if __name__ == '__main__':
    main() 