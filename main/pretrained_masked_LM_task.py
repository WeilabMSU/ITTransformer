import os
import argparse
import torch
from torch.utils.data import DataLoader
from itt.utils import LinearWarmupScheduler, CustomAtomInitializer, GaussianDistance
from itt.configuration_itt import IttConfig
from itt.modeling_itt_cross import IttForCrossMaskedLM
import numpy as np
import tarfile
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import shutil
import time
from datetime import timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='IT-Transformer Cross Masked LM Pretraining')
    parser.add_argument('--pretraining-data', type=str, required=True, help='Path to pretraining_data.tar containing .npz files')
    parser.add_argument('--label-csv', type=str, required=True, help='CSV file with columns [id,volume,atom_density,space_group,point_group,avg_electronegativity]')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--n-conv', type=int, default=4, help='layer number for graph part')
    parser.add_argument('--train-ratio', type=float, default=0.95, help='Train split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.04, help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.01, help='Test split ratio')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for splitting')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--standardize-features', action='store_true', help='Whether to standardize features (requires --feature-scaler)')
    parser.add_argument('--feature-scaler', type=str, default=None, help='Path to a saved feature scaler (joblib)')
    parser.add_argument('--mask-ratio', type=float, default=0.15, help='Ratio of part0 tokens to mask (default: 0.15)')
    return parser.parse_args()

class TarFeatureCrossMaskedDataset(torch.utils.data.Dataset):
    def __init__(self, tar_path, file_ids, feature_scaler=None, max_atom_number=256, n3=7, mask_ratio=0.15):
        self.tar_path = tar_path
        self.file_ids = file_ids
        self.feature_scaler = feature_scaler
        self.max_atom_number = max_atom_number
        self.n3 = n3
        self.mask_ratio = mask_ratio
        with tarfile.open(tar_path, 'r') as tar:
            self.members = {os.path.splitext(os.path.basename(m.name))[0]: m for m in tar.getmembers() if m.name.endswith('.npz')}
        self.atom_initializer = CustomAtomInitializer()
        self.dist_converter = GaussianDistance(dmin=0, dmax=8.0, step=0.2)
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        with tarfile.open(self.tar_path, 'r') as tar:
            member = self.members[file_id]
            f = tar.extractfile(member)
            data = np.load(f, allow_pickle=True)
            part0 = data['part0']
            part1 = data['part1']
            part2 = data['part2']
            part3_atom_nos = data['part3_atom_nos']
            part3_nbr_fea_dist = data['part3_nbr_fea_dist']
            part3_nbr_fea_idx = data['part3_nbr_fea_idx']
            part3_cluster_indices = data['part3_cluster_indices']
            
            if self.feature_scaler is not None:
                part0 = self.feature_scaler[0].transform(part0.reshape(-1, part0.shape[-1])).reshape(part0.shape)
                part1 = self.feature_scaler[1].transform(part1.reshape(-1, part1.shape[-1])).reshape(part1.shape)
                part2 = self.feature_scaler[2].transform(part2.reshape(-1, part2.shape[-1])).reshape(part2.shape)
            
            atom_fea = np.vstack([self.atom_initializer.get_atom_fea(no) for no in part3_atom_nos])
            nbr_fea = self.dist_converter.expand(part3_nbr_fea_dist.astype(np.float32)/1000)
            nbr_fea_idx = part3_nbr_fea_idx.astype(int)
            cluster_indices = part3_cluster_indices.astype(int)
            
            # Create mask for part0 features (mask 15% of the d0 features)
            mask = np.random.random(part0.shape) < self.mask_ratio  # Boolean array of same shape as part0
            # Actually mask part0 values (replace masked features with zeros)
            masked_part0 = part0.copy()
            masked_part0[mask] = 0.0  # Replace masked features with zeros
            
            return (masked_part0, part1, part2, (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, [])), (mask, part0)

def fn_collate_cross_masked(dataset_list):
    batch_part0, batch_part1, batch_part2 = [], [], []
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_cluster_indices = [], [], [], []
    batch_crystal_atom_idx = []
    masks = []
    target_part0 = []
    base_idx = 0
    
    for i, (data_point, labels) in enumerate(dataset_list):
        masked_part0, part1, part2, (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx) = data_point
        mask, part0_target = labels
        
        batch_part0.append(torch.from_numpy(masked_part0))
        batch_part1.append(torch.from_numpy(part1))
        batch_part2.append(torch.from_numpy(part2))
        
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(torch.from_numpy(atom_fea))
        batch_nbr_fea.append(torch.from_numpy(nbr_fea))
        batch_nbr_fea_idx.append(torch.from_numpy(nbr_fea_idx) + base_idx)
        batch_cluster_indices.append(torch.from_numpy(cluster_indices))
        
        if isinstance(crystal_atom_idx, list):
            batch_crystal_atom_idx.append(torch.LongTensor(crystal_atom_idx))
        else:
            batch_crystal_atom_idx.append(crystal_atom_idx)
        
        base_idx += n_i
        masks.append(torch.tensor(mask, dtype=torch.bool))
        target_part0.append(torch.from_numpy(part0_target).float())
    
    return (
        torch.stack(batch_part0, dim=0).float(),
        torch.stack(batch_part1, dim=0).float(),
        torch.stack(batch_part2, dim=0).float(),
        (
            torch.cat(batch_atom_fea, dim=0).float(),
            torch.cat(batch_nbr_fea, dim=0).float(),
            torch.cat(batch_nbr_fea_idx, dim=0).long(),
            torch.cat(batch_cluster_indices, dim=0).long(),
            batch_crystal_atom_idx
        )
    ), (torch.stack(masks, dim=0), torch.stack(target_part0, dim=0).float())

def split_ids(file_ids, train_ratio, val_ratio, test_ratio, seed=42):
    np.random.seed(seed)
    n = len(file_ids)
    idxs = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = [file_ids[i] for i in idxs[:n_train]]
    val_ids = [file_ids[i] for i in idxs[n_train:n_train+n_val]]
    test_ids = [file_ids[i] for i in idxs[n_train+n_val:]]
    return train_ids, val_ids, test_ids

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_config = state['config']
    config_dict = model_config.to_dict() if hasattr(model_config, 'to_dict') else model_config.__dict__
    state['config'] = config_dict
    torch.save(state, filename)
    if is_best:
        best_model_path = os.path.join(os.path.dirname(filename), 'pretrained_model_best.pth.tar')
        shutil.copyfile(filename, best_model_path)
        print(f"Model saved at epoch {state['epoch']}, saved to {best_model_path}")
        return state['state_dict']
    return None

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def train(train_loader, model, optimizer, scheduler, device, args):
    model.train()
    total_loss = 0.0
    total_count = 0
    
    for i, batch in enumerate(train_loader):
        (part0, part1, part2, (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)), labels = batch
        part0 = part0.to(device).float()
        part1 = part1.to(device).float()
        part2 = part2.to(device).float()
        atom_fea = atom_fea.to(device).float()
        nbr_fea = nbr_fea.to(device).float()
        
        masks, part0_target = labels
        masks = masks.to(device)
        part0_target = part0_target.to(device)
        
        graph_data = (atom_fea, nbr_fea, nbr_fea_idx.to(device), cluster_indices.to(device), crystal_atom_idx)
        
        # Create mask_info for part0
        mask_info = {'part0': (masks, part0_target)}  # (mask, target)
        
        optimizer.zero_grad()
        outputs = model(part0, part1, part2, graph_data, mask_info=mask_info)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * part0.shape[0]
        total_count += part0.shape[0]
        
        if i % args.print_freq == 0:
            print(f'Step {i}/{len(train_loader)} | Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / total_count
    return avg_loss

def validate(val_loader, model, device, args):
    model.eval()
    val_loss = 0.0
    val_masked_loss = 0.0
    val_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            (part0, part1, part2, (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)), labels = batch
            part0 = part0.to(device).float()
            part1 = part1.to(device).float()
            part2 = part2.to(device).float()
            atom_fea = atom_fea.to(device).float()
            nbr_fea = nbr_fea.to(device).float()
            
            masks, part0_target = labels
            masks = masks.to(device)
            part0_target = part0_target.to(device)
            
            graph_data = (atom_fea, nbr_fea, nbr_fea_idx.to(device), cluster_indices.to(device), crystal_atom_idx)
            mask_info = {'part0': (masks, part0_target)}
            
            outputs = model(part0, part1, part2, graph_data, mask_info=mask_info)
            loss = outputs['loss']
            loss_masked = outputs['masked_loss']
            
            batch_size = part0.shape[0]
            val_loss += loss.item() * batch_size
            val_masked_loss += loss_masked.item() * batch_size
            val_count += batch_size
    
    avg_val_loss = val_loss / val_count
    avg_val_masked_loss = val_masked_loss / val_count
    
    return {
        'val_loss': avg_val_loss,
        'val_masked_loss': avg_val_masked_loss
    }

def test(test_loader, model, device, args):
    model.eval()
    test_loss = 0.0
    test_masked_loss = 0.0
    test_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            (part0, part1, part2, (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)), labels = batch
            part0 = part0.to(device).float()
            part1 = part1.to(device).float()
            part2 = part2.to(device).float()
            atom_fea = atom_fea.to(device).float()
            nbr_fea = nbr_fea.to(device).float()
            
            masks, part0_target = labels
            masks = masks.to(device)
            part0_target = part0_target.to(device)
            
            graph_data = (atom_fea, nbr_fea, nbr_fea_idx.to(device), cluster_indices.to(device), crystal_atom_idx)
            mask_info = {'part0': (masks, part0_target)}
            
            outputs = model(part0, part1, part2, graph_data, mask_info=mask_info)
            loss = outputs['loss']
            loss_masked = outputs['masked_loss']
            
            batch_size = part0.shape[0]
            test_loss += loss.item() * batch_size
            test_masked_loss += loss_masked.item() * batch_size
            test_count += batch_size
    
    avg_test_loss = test_loss / test_count
    avg_test_masked_loss = test_masked_loss / test_count
    
    return {
        'test_loss': avg_test_loss,
        'test_masked_loss': avg_test_masked_loss
    }

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load label CSV (only need file IDs for this task)
    df = pd.read_csv(args.label_csv)
    file_ids = df['id'].tolist()

    # Split
    train_ids, val_ids, test_ids = split_ids(file_ids, args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed)

    # Feature scaler
    feature_scaler = None
    if args.standardize_features:
        if not args.feature_scaler:
            raise ValueError('--feature-scaler must be provided if --standardize-features is set.')
        feature_scaler = load(args.feature_scaler)
        print('✓ Feature scalers loaded')

    # Datasets and loaders
    # Use config defaults for max_atom_number and n3
    config = IttConfig(n_conv=args.n_conv)
    max_atom_number = getattr(config, 'max_atom_number', 264)
    n3 = getattr(config, 'n3', 7)
    
    train_dataset = TarFeatureCrossMaskedDataset(args.pretraining_data, train_ids, feature_scaler, max_atom_number, n3, args.mask_ratio)
    val_dataset = TarFeatureCrossMaskedDataset(args.pretraining_data, val_ids, feature_scaler, max_atom_number, n3, args.mask_ratio)
    test_dataset = TarFeatureCrossMaskedDataset(args.pretraining_data, test_ids, feature_scaler, max_atom_number, n3, args.mask_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=fn_collate_cross_masked, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=fn_collate_cross_masked, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=fn_collate_cross_masked, num_workers=args.num_workers)

    # Model
    model = IttForCrossMaskedLM(config, mask_ratio=args.mask_ratio).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    base_lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = min(1, int(0.05 * args.epochs)) * len(train_loader)
    scheduler = LinearWarmupScheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")

    print(model)
    best_val_loss = float('inf')
    best_model_state = None
    total_start_time = time.time()
    print("\nStarting training...")
    print(f"Total epochs: {args.epochs}")
    print(f"Total steps per epoch: {len(train_loader)}")
    print(f"Total training steps: {len(train_loader) * args.epochs}")
    print(f"Mask ratio: {args.mask_ratio}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train(train_loader, model, optimizer, scheduler, device, args)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f}")
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        avg_epoch_time = total_time / (epoch + 1)
        estimated_remaining_time = avg_epoch_time * (args.epochs - epoch - 1)
        print(f"Time: {format_time(epoch_time)} | Total: {format_time(total_time)} | ETA: {format_time(estimated_remaining_time)}")
        
        val_metrics = validate(val_loader, model, device, args)
        print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_metrics['val_loss']:.6f} | Masked Loss: {val_metrics['val_masked_loss']:.6f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6e}")
        
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': config,
            'feature_scaler': feature_scaler,
            'args': vars(args) if hasattr(args, '__dict__') else args,
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
        }
        torch.save(checkpoint_state, os.path.join(args.save_path, 'pretrained_model_last.pth.tar'))
        
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            returned_state = save_checkpoint(checkpoint_state, True, os.path.join(args.save_path, 'checkpoint.pth.tar'))
            if returned_state is not None:
                best_model_state = returned_state
    
    total_training_time = time.time() - total_start_time
    print(f"\nTraining completed! Total time: {format_time(total_training_time)}")
    
    if best_model_state is not None:
        print("Loading best model for testing...")
        model.load_state_dict(best_model_state)
        print("✓ Best model loaded for testing")
    
    test_metrics = test(test_loader, model, device, args)
    print(f"Test Loss: {test_metrics['test_loss']:.6f}")
    print(f"  Masked Loss: {test_metrics['test_masked_loss']:.6f}")

if __name__ == '__main__':
    main() 