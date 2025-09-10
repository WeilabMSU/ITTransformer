import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from joblib import load
import json
import glob
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.itt.utils import IndividualFeatureDataset, collate_fn
from main.itt.configuration_itt import IttConfig
from main.itt.modeling_itt_cross import IttCrossSequenceClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='IT-Transformer Attention Score Generation')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained finetuned model checkpoint')
    parser.add_argument('--features-folder', type=str, required=True, help='Path to folder containing .npz feature files')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save attention scores')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--standardize-features', action='store_true', help='Whether to standardize features (requires --feature-scaler)')
    parser.add_argument('--feature-scaler', type=str, default=None, help='Path to feature scaler (required if --standardize-features is set)')
    return parser.parse_args()

class AttentionExtractor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Reconstruct config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            if isinstance(config_dict, dict):
                config = IttConfig(**config_dict)
            else:
                config_data = config_dict.to_dict() if hasattr(config_dict, 'to_dict') else config_dict.__dict__
                config = IttConfig(**config_data)
        else:
            config = IttConfig()
        
        # Initialize model
        self.model = IttCrossSequenceClassifier(config).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # Store config for reference
        self.config = config
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Model config: n0={config.n0}, n1={config.n1}, n2={config.n2}")

    def extract_attention_scores(self, dataloader, file_ids):
        """
        Extract two types of attention scores:
        1. Self-attention scores from decoder layers [decoder_length, decoder_length] 
           (part0 + part1 + atoms attending to each other)
        2. Cross-attention scores between decoder and encoder [decoder_length, encoder_length]
           (decoder queries against encoder key-value pairs)
        
        Returns:
            - all_self_attention: List of self-attention scores for each sample
            - all_cross_attention: List of cross-attention scores for each sample
            - sample_ids: List of sample IDs
            - atomic_lengths: List of actual atomic lengths for each sample
        """
        all_self_attention = []
        all_cross_attention = []
        sample_ids = []
        atomic_lengths = []
        seen = 0
        
        # Hooks to capture attention scores
        self_attention_scores = []
        cross_attention_scores = []
        
        def self_attention_hook(module, input, output):
            # Capture self-attention scores from decoder layers
            # The output should be (context_layer, attention_probs)
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights = output[1].detach().cpu()  # attention_probs
            else:
                return
            
            # Ensure we have the right shape: [batch, num_heads, seq_len, seq_len]
            if attention_weights.dim() == 4:
                # Average across attention heads to get [batch, seq_len, seq_len]
                attention_weights_avg = torch.mean(attention_weights, dim=1)
                self_attention_scores.append(attention_weights_avg)
                print(f"Captured self-attention scores shape: {attention_weights_avg.shape}")
        
        def cross_attention_hook(module, input, output):
            # Capture cross-attention scores between decoder and encoder
            # The output should be (context_layer, attention_probs)
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights = output[1].detach().cpu()  # attention_probs
            else:
                return
            
            # Ensure we have the right shape: [batch, num_heads, seq_len, seq_len]
            if attention_weights.dim() == 4:
                # Average across attention heads to get [batch, seq_len, seq_len]
                attention_weights_avg = torch.mean(attention_weights, dim=1)
                cross_attention_scores.append(attention_weights_avg)
                print(f"Captured cross-attention scores shape: {attention_weights_avg.shape}")
        
        # Register hooks
        hooks = []
        
        # Hook for self-attention in decoder layers (IttDecoderLayer.self_attention)
        for name, module in self.model.named_modules():
            if 'self_attention' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(self_attention_hook)
                hooks.append(hook)
                print(f"Registered self-attention hook on: {name}")
        
        # Hook for cross-attention between decoder and encoder (IttCrossAttention)
        for name, module in self.model.named_modules():
            if 'cross_attention' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(cross_attention_hook)
                hooks.append(hook)
                print(f"Registered cross-attention hook on: {name}")
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    inputs, _ = batch
                    part0, part1, part2, graph_data_tuple = inputs
                    part0 = part0.to(self.device).float()
                    part1 = part1.to(self.device).float()
                    part2 = part2.to(self.device).float()
                    
                    atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx = graph_data_tuple
                    atom_fea = atom_fea.to(self.device).float()
                    nbr_fea = nbr_fea.to(self.device).float()
                    nbr_fea_idx = nbr_fea_idx.to(self.device)
                    cluster_indices = cluster_indices.to(self.device)
                    crystal_atom_idx = [idx.to(self.device) for idx in crystal_atom_idx]
                    graph_data = (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)
                    
                    # Clear previous attention scores
                    self_attention_scores.clear()
                    cross_attention_scores.clear()
                    
                    # Forward pass to trigger hooks
                    outputs = self.model(
                        input_part0=part0,
                        input_part1=part1,
                        input_part2=part2,
                        graph_data=graph_data,
                        use_cross_attention_mask=False
                    )
                    
                    # Store self-attention scores for this batch
                    if self_attention_scores:
                        # Use the last captured self-attention scores (from the final decoder layer)
                        batch_self_attention = self_attention_scores[-1].numpy()
                        all_self_attention.append(batch_self_attention)
                        print(f"Self-attention scores shape: {batch_self_attention.shape}")
                    else:
                        # Create dummy self-attention scores [decoder_length, decoder_length]
                        batch_size = part0.shape[0]
                        # Calculate decoder length: n0 + n1 + max_atom_number
                        decoder_len = self.config.n0 + self.config.n1 + getattr(self.config, 'max_atom_number', 256)
                        dummy_scores = np.eye(decoder_len)
                        dummy_scores = np.tile(dummy_scores, (batch_size, 1, 1))
                        all_self_attention.append(dummy_scores)
                        print(f"Warning: No self-attention scores captured, using dummy scores")
                    
                    # Store cross-attention scores for this batch
                    if cross_attention_scores:
                        # Use the last captured cross-attention scores (from the final decoder layer)
                        batch_cross_attention = cross_attention_scores[-1].numpy()
                        all_cross_attention.append(batch_cross_attention)
                        print(f"Cross-attention scores shape: {batch_cross_attention.shape}")
                    else:
                        # Create dummy cross-attention scores [decoder_length, encoder_length]
                        batch_size = part0.shape[0]
                        # Calculate decoder and encoder lengths
                        decoder_len = self.config.n0 + self.config.n1 + getattr(self.config, 'max_atom_number', 256)
                        encoder_len = self.config.n2  # part2 length
                        dummy_scores = np.ones((batch_size, decoder_len, encoder_len)) / encoder_len
                        all_cross_attention.append(dummy_scores)
                        print(f"Warning: No cross-attention scores captured, using dummy scores")
                    
                    batch_size = part0.shape[0]
                    sample_ids.extend(file_ids[seen:seen+batch_size])
                    
                    # Record actual atomic lengths for each sample in the batch
                    for i in range(batch_size):
                        # Get the actual number of atoms for this sample
                        # crystal_atom_idx[i] contains the atom indices for sample i
                        actual_atom_length = len(crystal_atom_idx[i])
                        atomic_lengths.append(actual_atom_length)
                    
                    seen += batch_size
                    
                    if batch_idx % 10 == 0:
                        print(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Concatenate all attention scores
        if all_self_attention:
            all_self_attention = np.concatenate(all_self_attention, axis=0)
        else:
            all_self_attention = np.array([])
        
        if all_cross_attention:
            all_cross_attention = np.concatenate(all_cross_attention, axis=0)
        else:
            all_cross_attention = np.array([])
        
        return all_self_attention, all_cross_attention, sample_ids, atomic_lengths

def create_dataloader(features_folder, batch_size, num_workers, feature_scaler=None):
    """Create dataloader from features folder."""
    # Get all .npz files in the folder and sort them for consistent ordering
    feature_files = glob.glob(os.path.join(features_folder, "*.npz"))
    feature_files.sort()  # Sort for consistent ordering
    
    # Extract file IDs (remove .npz extension and folder path)
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in feature_files]
    indices = np.arange(len(file_ids))
    values = np.zeros(len(file_ids), dtype=np.float32)  # Default to zeros
    
    dataset = IndividualFeatureDataset((indices, file_ids, values), features_folder, feature_scaler)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    return dataloader, file_ids

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize attention extractor
    extractor = AttentionExtractor(args.model_path, args.device)
    
    # Load feature scaler if needed
    feature_scaler = None
    if args.standardize_features:
        if not args.feature_scaler:
            raise ValueError('--feature-scaler must be provided if --standardize-features is set')
        if not os.path.exists(args.feature_scaler):
            raise FileNotFoundError(f'Feature scaler file not found: {args.feature_scaler}')
        print(f'Loading feature scaler from: {args.feature_scaler}')
        feature_scaler = load(args.feature_scaler)
        print('✓ Feature scaler loaded')
    
    # Create dataloader
    print(f"Loading features from: {args.features_folder}")
    dataloader, file_ids = create_dataloader(args.features_folder, args.batch_size, args.num_workers, feature_scaler)
    print(f"Created dataloader with {len(dataloader)} batches")
    print(f"Processing {len(file_ids)} samples")
    print(f"Sample file IDs: {file_ids[:5]}...")
    
    # Extract attention scores
    print("Extracting attention scores...")
    self_attention_scores, cross_attention_scores, sample_ids, atomic_lengths = extractor.extract_attention_scores(dataloader, file_ids)
    
    print(f"Self-attention scores shape: {self_attention_scores.shape}")
    print(f"Cross-attention scores shape: {cross_attention_scores.shape}")
    
    # Save attention scores
    attention_file = os.path.join(args.output_path, "attention_scores.npz")
    np.savez(attention_file, 
            self_attention_scores=self_attention_scores,
            cross_attention_scores=cross_attention_scores,
            sample_ids=sample_ids,
            atomic_lengths=atomic_lengths)
    print(f"✓ Attention scores saved to {attention_file}")
    
    # Save metadata
    metadata = {
        'model_path': args.model_path,
        'features_folder': args.features_folder,
        'num_samples': len(sample_ids),
        'self_attention_shape': self_attention_scores.shape,
        'cross_attention_shape': cross_attention_scores.shape,
        'atomic_lengths_stats': {
            'min': min(atomic_lengths),
            'max': max(atomic_lengths),
            'mean': np.mean(atomic_lengths),
            'std': np.std(atomic_lengths)
        },
        'config': {
            'n0': extractor.config.n0,
            'n1': extractor.config.n1,
            'n2': extractor.config.n2,
            'max_atom_number': getattr(extractor.config, 'max_atom_number', 256)
        }
    }
    
    metadata_file = os.path.join(args.output_path, "attention_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {metadata_file}")
    print(f"✓ Processed {len(sample_ids)} samples")
    print(f"✓ Self-attention matrix shape: {self_attention_scores.shape}")
    print(f"✓ Cross-attention matrix shape: {cross_attention_scores.shape}")
    print(f"✓ Atomic lengths - Min: {min(atomic_lengths)}, Max: {max(atomic_lengths)}, Mean: {np.mean(atomic_lengths):.1f}")

if __name__ == '__main__':
    main()
