import os
import json
import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import glob

from itt.configuration_itt import IttConfig
from itt.modeling_itt_cross import IttCrossSequenceClassifier
from itt.utils import (
    calculate_metrics,
    calculate_classification_metrics,
    IndividualFeatureDataset,
    collate_fn,
    AverageMeter,
)

def parse_args():
    parser = argparse.ArgumentParser(description='IT-Transformer Cross-Attention Prediction')
    
    # Data related arguments
    parser.add_argument('--features-folder', type=str, required=True,
                      help='path to folder containing pre-calculated feature .npz files (one per sample)')
    parser.add_argument('--label-csv', type=str, required=True,
                      help='path to CSV file containing file IDs and labels (labels can be any values)')
    
    # Model loading options
    parser.add_argument('--model-paths', type=str, nargs='+', required=True,
                      help='list of model checkpoint paths (single path for individual prediction, multiple paths for ensemble prediction)')
    
    # Task related arguments
    parser.add_argument('--task', choices=['regression', 'classification'],
                      default='regression', help='task type')
    parser.add_argument('--num-labels', type=int, default=1,
                      help='number of labels for classification')
    
    
    # Model related arguments (only used if config not found in checkpoint)
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
    
    # Feature scaler argument (always required for feature normalization)
    parser.add_argument('--feature-scaler', type=str, required=True,
                      help='path to feature scaler file (required for feature normalization)')
    
    # Output arguments
    parser.add_argument('--output-csv', type=str, default=None,
                      help='path to save prediction results (default: predictions.csv in current directory)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='batch size for prediction')
    parser.add_argument('--num-workers', type=int, default=2, 
                      help='Number of workers for data loading (default: 2)')
    
    return parser.parse_args()


def load_model_from_checkpoint(model_path, device, args):
    """
    Load a single model from checkpoint file.
    
    Args:
        model_path: Path to model checkpoint file
        device: Device to load model on
        args: Command line arguments
        
    Returns:
        Tuple of (model, config, label_scaler, feature_scaler, label_mapping_info, saved_args)
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        if isinstance(config_dict, dict):
            config = IttConfig(**config_dict)
        else:
            config = IttConfig.to_dict(config_dict)
        print(f"Loaded config from checkpoint")
    else:
        # Create config from command line arguments
        print(f"No config found in checkpoint, using command line arguments")
        config = IttConfig(
            n0=args.n0, d0=args.d0,
            n1=args.n1, d1=args.d1,
            n2=args.n2, d2=args.d2,
            classifier_head_method=args.classifier_head_method,
            num_labels=args.num_labels,
            orig_atom_fea_len=92,  # Default from atom_init.json
            nbr_fea_len=41,  # Default from GaussianDistance
            n_conv=args.n_conv,
        )
    
    # Load scalers and other saved information
    label_scaler = checkpoint.get('label_scaler', None)
    feature_scaler = checkpoint.get('feature_scaler', None)
    label_mapping_info = checkpoint.get('label_mapping_info', None)
    saved_args = checkpoint.get('args', None)
    
    if label_scaler is not None:
        print(f"  ✓ Loaded label scaler from checkpoint")
    if feature_scaler is not None:
        print(f"  ✓ Loaded feature scaler from checkpoint")
    if label_mapping_info is not None:
        print(f"  ✓ Loaded label mapping info from checkpoint")
    if saved_args is not None:
        print(f"  ✓ Loaded saved training arguments from checkpoint")
    
    # Create model
    model = IttCrossSequenceClassifier(config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"  ✓ Model loaded successfully")
    
    return model, config, label_scaler, feature_scaler, label_mapping_info, saved_args

def load_models_for_ensemble(model_paths, device, args):
    """
    Load multiple models for ensemble prediction.
    
    Args:
        model_paths: List of specific model paths
        device: Device to load models on
        args: Command line arguments
        
    Returns:
        Tuple of (models, config, label_scaler, feature_scaler, label_mapping_info, saved_args)
    """
    model_files = model_paths
    print(f"Loading {len(model_files)} specific models:")
    
    for f in sorted(model_files):
        print(f"  {os.path.basename(f)}")
    
    # Load first model to get config and scalers
    print(f"\nLoading first model to get config and scalers...")
    first_model, config, label_scaler, feature_scaler, label_mapping_info, saved_args = load_model_from_checkpoint(model_files[0], device, args)
    
    if label_scaler is None:
        print(f"  !!!! Warning: The first model should be checkpoint.tar.gz/model_best.pth.tar to get the [label scaler] !!!!!")

    # Load all models
    models = [first_model]
    for i, model_file in enumerate(model_files[1:], 1):
        print(f"\nLoading model {i+1}/{len(model_files)}: {os.path.basename(model_file)}")
        model, _, _, _, _, _ = load_model_from_checkpoint(model_file, device, args)
        models.append(model)
    
    print(f"\n✓ Loaded {len(models)} models for ensemble prediction")
    
    return models, config, label_scaler, feature_scaler, label_mapping_info, saved_args

def predict_and_save_single(model, test_loader, test_file_ids, original_labels, device, args, label_scaler):
    """
    Predict using single model and save results.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        test_file_ids: List of file IDs for test samples
        original_labels: Original labels from CSV (for output only)
        device: Device to run evaluation on
        args: Command line arguments
        label_scaler: Label scaler if labels were standardized
    """
    print(f"\nPredicting with single model...")
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch  # Ignore labels during prediction
            
            part0, part1, part2, graph_data_tuple = inputs
            # Convert input tensors to float32 before moving to device
            part0 = part0.to(dtype=torch.float32).to(device)
            part1 = part1.to(dtype=torch.float32).to(device)
            part2 = part2.to(dtype=torch.float32).to(device)
            
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
                          graph_data=graph_data, use_cross_attention_mask=False)
            predictions = outputs[0]
            
            all_preds.append(predictions.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    
    # Prepare predictions for output
    if args.task == 'classification':
        # For classification: convert probabilities to predicted class labels
        predicted_classes = np.argmax(all_preds, axis=1)  # [total_samples]
        predictions_df = pd.DataFrame({
            'file_id': test_file_ids,
            'original_label': original_labels,
            'predicted_class': predicted_classes,
        })
    else:
        # For regression: use original scale predictions
        if label_scaler is not None:
            # Transform predictions back to original scale
            orig_preds = label_scaler.inverse_transform(all_preds)
            predictions_df = pd.DataFrame({
                'file_id': test_file_ids,
                'original_label': original_labels,
                'predicted': orig_preds.flatten()
            })
        else:
            predictions_df = pd.DataFrame({
                'file_id': test_file_ids,
                'original_label': original_labels,
                'predicted': all_preds.flatten()
            })
    
    # Save predictions (no header as requested)
    output_path = args.output_csv if args.output_csv else 'predictions.csv'
    predictions_df.to_csv(output_path, index=False, header=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Print column descriptions
    print("\nColumn descriptions:")
    print("  Column 1: file_id (original file identifier)")
    print("  Column 2: original_label (from input CSV)")
    if args.task == 'classification':
        print("  Column 3: predicted_class (predicted class)")
    else:
        print("  Column 3: predicted (predicted value in original scale)")

def predict_and_save_ensemble(models, test_loader, test_file_ids, original_labels, device, args, label_scaler):
    """
    Predict using ensemble of models and save results.
    
    Args:
        models: List of trained models
        test_loader: DataLoader for test data
        test_file_ids: List of file IDs for test samples
        original_labels: Original labels from CSV (for output only)
        device: Device to run evaluation on
        args: Command line arguments
        label_scaler: Label scaler if labels were standardized
    """
    print(f"\nPredicting with ensemble of {len(models)} models...")
    
    for model in models:
        model.eval()
    
    all_ensemble_preds = []
    
    # Store individual model predictions
    individual_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch  # Ignore labels during prediction
            
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
            for model in models:
                outputs = model(input_part0=part0, input_part1=part1, input_part2=part2, 
                              graph_data=graph_data, use_cross_attention_mask=False)
                predictions = outputs[0]
                batch_preds.append(predictions.cpu().numpy())
            
            # Average predictions from all models
            batch_preds = np.array(batch_preds)  # [num_models, batch_size, num_classes] for classification, [num_models, batch_size, 1] for regression
            ensemble_pred = np.mean(batch_preds, axis=0)  # [batch_size, num_classes] and [batch_size, 1] for regression
            
            all_ensemble_preds.append(ensemble_pred)
            individual_preds.append(batch_preds.transpose(1, 0, 2))  # [batch_size, num_models, ...]
    
    all_ensemble_preds = np.concatenate(all_ensemble_preds, axis=0)
    individual_preds = np.concatenate(individual_preds, axis=0)  # [total_samples, num_models, ...]
    individual_preds = individual_preds.transpose(1, 0, 2)  # [num_models, total_samples, ...]
    
    # Prepare predictions for output
    if args.task == 'classification':
        # For classification: convert ensemble probabilities to predicted class labels
        ensemble_predicted_classes = np.argmax(all_ensemble_preds, axis=1)  # [total_samples]
        
        predictions_df = pd.DataFrame({
            'file_id': test_file_ids,
            'original_label': original_labels,
            'ensemble_predicted_class': ensemble_predicted_classes,
        })
        
        # Add individual model predictions
        for i in range(len(models)):
            col_name = f'model_{i+1}_predicted_class'
            predicted_classes = np.argmax(individual_preds[i], axis=1)  # [total_samples]
            predictions_df[col_name] = predicted_classes
        
    else:
        # For regression: use original scale predictions
        if label_scaler is not None:
            # Transform ensemble predictions back to original scale
            orig_ensemble_preds = label_scaler.inverse_transform(all_ensemble_preds)
            predictions_df = pd.DataFrame({
                'file_id': test_file_ids,
                'original_label': original_labels,
                'ensemble_predicted': orig_ensemble_preds.flatten()
            })
            
            # Add individual model predictions (in original scale)
            for i in range(len(models)):
                col_name = f'model_{i+1}_predicted'
                individual_orig_preds = label_scaler.inverse_transform(individual_preds[i])
                predictions_df[col_name] = individual_orig_preds.flatten()
        else:
            predictions_df = pd.DataFrame({
                'file_id': test_file_ids,
                'original_label': original_labels,
                'ensemble_predicted': all_ensemble_preds.flatten()
            })
            
            # Add individual model predictions
            for i in range(len(models)):
                col_name = f'model_{i+1}_predicted'
                predictions_df[col_name] = individual_preds[i].flatten()
    
    # Save predictions (no header as requested)
    output_path = args.output_csv if args.output_csv else 'predictions_ensemble.csv'
    predictions_df.to_csv(output_path, index=False, header=False)
    print(f"\nEnsemble predictions saved to: {output_path}")
    
    # Print column descriptions
    print("\nColumn descriptions:")
    print("  Column 1: file_id (original file identifier)")
    print("  Column 2: original_label (from input CSV)")
    if args.task == 'classification':
        print("  Column 3: ensemble_predicted_class (ensemble predicted class)")
        for i in range(len(models)):
            print(f"  Column {4+i}: model_{i+1}_predicted_class (individual model {i+1} predicted class)")
    else:
        print("  Column 3: ensemble_predicted (ensemble predicted value in original scale)")
        for i in range(len(models)):
            print(f"  Column {4+i}: model_{i+1}_predicted (individual model {i+1} predicted value in original scale)")

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

    # Load label data
    print(f"Loading labels from: {args.label_csv}")
    df = pd.read_csv(args.label_csv, header=None)
    file_ids = df[0].values  # First column contains file IDs
    original_labels = df[1].values  # Second column contains labels (for output only)
    
    print(f"Total samples: {len(file_ids)}")
    
    # Verify that all label IDs have corresponding feature files
    print(f"Verifying feature files exist for all {len(file_ids)} samples...")
    missing_files = []
    for file_id in file_ids:
        feature_file = os.path.join(args.features_folder, f"{file_id}.npz")
        if not os.path.exists(feature_file):
            missing_files.append(file_id)
    
    if missing_files:
        raise ValueError(f"Missing feature files for IDs: {missing_files[:10]}... (showing first 10)")
    
    print("✓ All feature files found")
    
    # Load models
    if len(args.model_paths) == 1:
        # Single model
        print(f"\nLoading single model...")
        model, config, label_scaler, feature_scaler, label_mapping_info, saved_args = load_model_from_checkpoint(args.model_paths[0], device, args)
        models = [model]
        is_ensemble = False
    else:
        # Ensemble models
        print(f"\nLoading ensemble of {len(args.model_paths)} models...")
        models, config, label_scaler, feature_scaler, label_mapping_info, saved_args = load_models_for_ensemble(
            args.model_paths, device, args
        )
        is_ensemble = True
    
    # Print information about loaded configuration
    print(f"\nLoaded model configuration:")
    print(f"  Task: {args.task}")
    print(f"  Classifier method: {config.classifier_head_method}")
    print(f"  Number of labels: {config.num_labels}")
    if label_scaler is not None:
        print(f"  Label scaler: Available")
    else:
        print(f"  Label scaler: None")
    if label_mapping_info is not None:
        print(f"  Label mapping: Available")
        print(f"    Original labels: {label_mapping_info['original_labels']}")
        print(f"    Mapped to: {label_mapping_info['mapped_labels']}")
    else:
        print(f"  Label mapping: None")
    
    # Load feature scaler
    print(f"\nLoading feature scaler from: {args.feature_scaler}")
    if not os.path.exists(args.feature_scaler):
        raise FileNotFoundError(f"Feature scaler file not found: {args.feature_scaler}")
    feature_scaler = load(args.feature_scaler)
    print("✓ Feature scaler loaded")
    
    # Create dataset and data loader
    print(f"\nCreating data loader...")
    # Create dummy split data (all samples are test samples, labels are ignored during prediction)
    dummy_labels = np.zeros(len(file_ids))  # Dummy labels, not used during prediction
    test_split_data = (np.arange(len(file_ids)), file_ids, dummy_labels)
    test_dataset = IndividualFeatureDataset(test_split_data, args.features_folder, feature_scaler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    

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
    
    # Perform prediction
    print(f"\nStarting prediction...")
    
    if is_ensemble:
        predict_and_save_ensemble(
            models, test_loader, file_ids, original_labels, device, args, label_scaler
        )
    else:
        predict_and_save_single(
            models[0], test_loader, file_ids, original_labels, device, args, label_scaler
        )
    
    print(f"\nPrediction completed!")

if __name__ == '__main__':
    main()
