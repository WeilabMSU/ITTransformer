import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from itt.utils import IndividualFeatureDataset, collate_fn
from itt.configuration_itt import IttConfig
from itt.modeling_itt_cross import IttCrossSequenceClassifier, IttForCrossJointPredictionLM, IttForCrossMaskedAndAtomLM, IttForCrossMaskedLM
from joblib import load
import json

def parse_args():
    parser = argparse.ArgumentParser(description='IT-Transformer Cross-Attention Embedding Generation')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--features-folder', type=str, required=True, help='Path to folder of .npz feature files')
    parser.add_argument('--label-csv', type=str, default=None, help='Path to CSV file containing file IDs (must match order for embedding output). If None, will use default order of feature files in folder.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save embeddings')
    parser.add_argument('--embedding-type', type=str, choices=['cls', 'mean', 'split_mean'], 
                       default='cls', help='Type of pooling method for pre-classifier embeddings')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of DataLoader workers')
    parser.add_argument('--standardize-features', action='store_true', help='Whether to standardize features (requires --feature-scaler)')
    parser.add_argument('--feature-scaler', type=str, default=None, help='Path to feature scaler (required if --standardize-features is set)')
    parser.add_argument('--model-type', type=str, choices=['pretrained', 'finetuned'], 
                       default='finetuned', help='Type of model to load (pretrained or finetuned)')
    return parser.parse_args()

class EmbeddingExtractor:
    def __init__(self, model_path, model_type='finetuned', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        # Reconstruct config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            if isinstance(config_dict, dict):
                config = IttConfig(**config_dict)
            else:
                # Handle case where config is an IttConfig object
                config_data = config_dict.to_dict() if hasattr(config_dict, 'to_dict') else config_dict.__dict__
                config = IttConfig(**config_data)
        else:
            config = IttConfig()
        
        # Initialize model based on type
        if model_type == 'finetuned':
            # For finetuned models, use the sequence classifier
            self.model = IttCrossSequenceClassifier(config).to(self.device)
        elif model_type == 'pretrained':
            # For pretrained models, use the masked LM model from pretrained_masked_LM_task_cross.py
            self.model = IttForCrossMaskedLM(config).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"Model loaded from {model_path}")
        print(f"Model type: {model_type}")
        print(f"Device: {self.device}")

    def extract_embeddings(self, dataloader, file_ids, pooling_method):
        embeddings = []
        sample_ids = []
        seen = 0
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
                
                # Get model outputs using cross-attention interface
                if self.model_type == 'finetuned':
                    # For finetuned models, use the classifier interface but get pre-classifier embeddings
                    if hasattr(self.model, 'itt_cross'):
                        # Get the raw sequence output from the cross model
                        outputs = self.model.itt_cross(
                            input_part0=part0,
                            input_part1=part1,
                            input_part2=part2,
                            graph_data=graph_data,
                            use_cross_attention_mask=False
                        )
                        sequence_output = outputs[0]
                    else:
                        raise AttributeError("Finetuned model does not have itt_cross attribute")
                else:
                    # For pretrained models, use the model directly
                    outputs = self.model.itt_cross(
                        input_part0=part0,
                        input_part1=part1,
                        input_part2=part2,
                        graph_data=graph_data,
                        use_cross_attention_mask=False
                    )
                    sequence_output = outputs[0]
                
                # Apply the specified pooling method
                pooled_embeddings = self._apply_pooling_method(sequence_output, part0, part1, part2, pooling_method)
                embeddings.append(pooled_embeddings.cpu().numpy())
                batch_size = pooled_embeddings.shape[0]
                sample_ids.extend(file_ids[seen:seen+batch_size])
                seen += batch_size
                if batch_idx % 10 == 0:
                    print(f"Processed batch {batch_idx}/{len(dataloader)}")
        all_embeddings = np.concatenate(embeddings, axis=0)
        return all_embeddings, sample_ids



    def _apply_pooling_method(self, sequence_output, part0, part1, part2, method):
        """Apply pooling method to get embeddings from sequence output."""
        if method == "cls":
            # Use the first token ([CLS] token)
            return sequence_output[:, 0]  # [batch_size, hidden_size]
        
        elif method == "mean":
            # Average all tokens in the last decoder layer output, excluding zero-padded atomic embeddings
            # The sequence structure is: [part0, part1, atoms (padded to max_atom_number=256), part2]
            # We need to exclude the zero-padded atomic positions that don't represent real atoms
            
            # Create a mask for non-zero embeddings to exclude padding
            # Use a higher threshold to distinguish real embeddings from zero padding
            embedding_norms = torch.norm(sequence_output, dim=-1)  # [batch_size, seq_len]
            mask = embedding_norms > 1e-6  # [batch_size, seq_len] - exclude zero-padded positions
            
            # Apply mask and compute mean over only the real (non-padded) tokens
            masked_sequence = sequence_output * mask.unsqueeze(-1)  # [batch_size, seq_len, hidden_size]
            sum_embeddings = torch.sum(masked_sequence, dim=1)  # [batch_size, hidden_size]
            num_nonzero_tokens = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]
            num_nonzero_tokens = torch.clamp(num_nonzero_tokens, min=1)  # Avoid division by zero
            
            return sum_embeddings / num_nonzero_tokens  # [batch_size, hidden_size]
        
        elif method == "split_mean":
            # Average each part separately and concatenate
            n0 = self.model.config.n0
            n1 = self.model.config.n1
            n2 = self.model.config.n2
            max_atom_number = getattr(self.model.config, 'max_atom_number', 100)
            
            # Split sequence output by parts
            part0_emb = sequence_output[:, :n0, :]  # [batch_size, n0, hidden_size]
            part1_emb = sequence_output[:, n0:n0+n1, :]  # [batch_size, n1, hidden_size]
            atom_emb = sequence_output[:, n0+n1:n0+n1+max_atom_number, :]  # [batch_size, max_atom_number, hidden_size]
            part2_emb = sequence_output[:, n0+n1+max_atom_number:, :]  # [batch_size, n2, hidden_size]
            
            # Pool each part
            part0_pooled = torch.mean(part0_emb, dim=1)  # [batch_size, hidden_size]
            
            # For part1, use mask
            part1_input_mask = (torch.norm(part1, dim=-1) > 1e-8)
            masked_part1 = part1_emb * part1_input_mask.unsqueeze(-1)
            sum_part1 = torch.sum(masked_part1, dim=1)
            num_nonzero_part1 = torch.sum(part1_input_mask, dim=1, keepdim=True)
            num_nonzero_part1 = torch.clamp(num_nonzero_part1, min=1)
            part1_pooled = sum_part1 / num_nonzero_part1  # [batch_size, hidden_size]
            
            # For atoms, assume all are valid (could be improved with actual atom counts)
            atom_pooled = torch.mean(atom_emb, dim=1)  # [batch_size, hidden_size]
            
            # For part2, use mask
            part2_input_mask = (torch.norm(part2, dim=-1) > 1e-8)
            masked_part2 = part2_emb * part2_input_mask.unsqueeze(-1)
            sum_part2 = torch.sum(masked_part2, dim=1)
            num_nonzero_part2 = torch.sum(part2_input_mask, dim=1, keepdim=True)
            num_nonzero_part2 = torch.clamp(num_nonzero_part2, min=1)
            part2_pooled = sum_part2 / num_nonzero_part2  # [batch_size, hidden_size]
            
            # Concatenate all parts
            return torch.cat([part0_pooled, part1_pooled, atom_pooled, part2_pooled], dim=1)  # [batch_size, 4*hidden_size]
        
        else:
            raise ValueError(f"Unknown pooling method: {method}. Supported methods: ['cls', 'mean', 'split_mean']")

def create_dataloader(features_folder, file_ids, batch_size, num_workers, feature_scaler=None, values=None):
    indices = np.arange(len(file_ids))
    if values is None:
        values = np.zeros(len(file_ids), dtype=np.float32)
    dataset = IndividualFeatureDataset((indices, file_ids, values), features_folder, feature_scaler)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    return dataloader

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load file IDs - either from CSV or from folder
    if args.label_csv is not None:
        # Load file IDs from label CSV (first column)
        df = pd.read_csv(args.label_csv, header=None)
        file_ids = df[0].astype(str).tolist()
        indices = np.arange(len(file_ids))
        # If a second column exists, use it as values; otherwise, use zeros
        if df.shape[1] > 1:
            # Handle different data types (float, int, string) for labels
            raw_values = df[1].values
            try:
                # Try to convert to float first
                values = raw_values.astype(np.float32)
                print(f"Labels detected as numeric: {type(raw_values[0])}")
            except (ValueError, TypeError):
                # If conversion fails, treat as string labels for classification
                print(f"Labels detected as non-numeric: {type(raw_values[0])} - converting to class indices")
                print(f"Sample original labels: {raw_values[:5]}...")
                
                # Get unique labels and sort them for consistent mapping
                unique_labels = np.unique(raw_values)
                sorted_unique_labels = np.sort(unique_labels)
                
                # Create mapping from original labels to indices [0, 1, 2, ...]
                label_mapping = {original_label: idx for idx, original_label in enumerate(sorted_unique_labels)}
                
                # Convert string labels to indices
                values = np.array([label_mapping[label] for label in raw_values])
                
                print(f"Found {len(unique_labels)} unique classes")
                print(f"Label mapping: {label_mapping}")
                print(f"Sample converted labels: {values[:5]}...")
                
                # Save label mapping info for reference
                label_mapping_info = {
                    'num_classes': len(unique_labels),
                    'original_labels': sorted_unique_labels.tolist(),
                    'mapped_labels': list(range(len(unique_labels))),
                    'mapping': {str(k): int(v) for k, v in label_mapping.items()}
                }
        else:
            values = np.zeros(len(file_ids), dtype=np.float32)
            label_mapping_info = None
    else:
        # Use default order of feature files in folder
        print(f"No label CSV provided. Using default order of feature files in: {args.features_folder}")
        
        # Get all .npz files in the folder and sort them for consistent ordering
        import glob
        feature_files = glob.glob(os.path.join(args.features_folder, "*.npz"))
        feature_files.sort()  # Sort for consistent ordering
        
        # Extract file IDs (remove .npz extension and folder path)
        file_ids = [os.path.splitext(os.path.basename(f))[0] for f in feature_files]
        indices = np.arange(len(file_ids))
        values = np.zeros(len(file_ids), dtype=np.float32)  # Default to zeros
        label_mapping_info = None
        
        print(f"Found {len(file_ids)} feature files")
        print(f"Sample file IDs: {file_ids[:5]}...")
    # Check that all files exist (only needed when using CSV, since glob already verified files exist)
    if args.label_csv is not None:
        missing = [fid for fid in file_ids if not os.path.exists(os.path.join(args.features_folder, f"{fid}.npz"))]
        if missing:
            raise FileNotFoundError(f"Missing feature files for IDs: {missing[:10]}... (showing first 10)")
    extractor = EmbeddingExtractor(args.model_path, args.model_type, args.device)
    feature_scaler = None
    if args.standardize_features:
        if not args.feature_scaler:
            raise ValueError('--feature-scaler must be provided if --standardize-features is set. Use the utility in main/itt/utils.py to create one.')
        if not os.path.exists(args.feature_scaler):
            raise FileNotFoundError(f'Feature scaler file not found: {args.feature_scaler}')
        print(f'Loading feature scaler from: {args.feature_scaler}')
        feature_scaler = load(args.feature_scaler)
        print('✓ Feature scalers loaded')
    print(f"Loading features from {args.features_folder}")
    dataloader = create_dataloader(args.features_folder, file_ids, args.batch_size, args.num_workers, feature_scaler, values)
    print(f"Created dataloader with {len(dataloader)} batches")
    print(f"Extracting embeddings using {args.embedding_type} pooling method...")
    
    # Extract embeddings using the specified pooling method
    embeddings, sample_ids = extractor.extract_embeddings(dataloader, file_ids, args.embedding_type)
    embedding_shape = embeddings.shape
    print(f"Embeddings shape: {embedding_shape}")
    
    output_file = os.path.join(args.output_path, f"{args.embedding_type}_embeddings.npz")
    
    # Save embeddings, sample_ids, and labels (if available) to npz file
    save_data = {
        'embeddings': embeddings,
        'sample_ids': sample_ids,
        'labels': values,
    }
    
    np.savez(output_file, **save_data)
    metadata = {
        'embedding_type': args.embedding_type,
        'model_path': args.model_path,
        'model_type': args.model_type,
        'features_folder': args.features_folder,
        'label_csv': args.label_csv,
        'use_default_file_order': args.label_csv is None,
        'embedding_shape': embedding_shape,
        'num_samples': len(sample_ids),
        'batch_size': args.batch_size,
        'device': args.device,
        'standardize_features': args.standardize_features,
        'label_mapping_info': label_mapping_info if 'label_mapping_info' in locals() else None
    }
    metadata_file = os.path.join(args.output_path, f"{args.embedding_type}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Embeddings saved to {output_file}")
    print(f"✓ Metadata saved to {metadata_file}")
    print(f"✓ Extracted {len(sample_ids)} samples with shape {embedding_shape}")

if __name__ == '__main__':
    main() 