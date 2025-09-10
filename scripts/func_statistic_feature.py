import os
import tarfile
import numpy as np
import argparse
import tempfile
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Statistic part3_atom_nos length from .npz files in a tar archive')
    parser.add_argument('--tar-file', type=str, required=True, help='Path to the tar file containing .npz files')
    parser.add_argument('--output-hist', type=str, default='part3_atom_nos_length_hist.png', help='Output path for histogram PNG')
    return parser.parse_args()


def main():
    args = parse_args()
    lengths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(args.tar_file, 'r') as tar:
            npz_files = [m for m in tar.getmembers() if m.name.endswith('.npz')]
            print(f'Found {len(npz_files)} .npz files in the tar archive.')
            for member in npz_files:
                tar.extract(member, path=tmpdir)
                npz_path = os.path.join(tmpdir, member.name)
                try:
                    data = np.load(npz_path)
                    if 'part3_atom_nos' in data:
                        arr = data['part3_atom_nos']
                        lengths.append(len(arr))
                    else:
                        print(f"Warning: 'part3_atom_nos' not found in {member.name}")
                except Exception as e:
                    print(f"Error loading {member.name}: {e}")
    if not lengths:
        print('No part3_atom_nos data found.')
        return
    lengths = np.array(lengths)
    print(f"Total files with 'part3_atom_nos': {len(lengths)}")
    avg = np.mean(lengths)
    maxv = np.max(lengths)
    minv = np.min(lengths)
    medv = np.median(lengths)
    print(f"Average length: {avg:.2f}")
    print(f"Max length: {maxv}")
    print(f"Min length: {minv}")
    print(f"Median length: {medv}")
    # Plot histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title('Histogram of part3_atom_nos Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add statistics text on the right side
    stats_text = f"Max: {maxv}\nMin: {minv}\nMean: {avg:.2f}\nMedian: {medv}"
    plt.gca().text(1.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(args.output_hist)
    print(f"Histogram saved to {args.output_hist}")


if __name__ == '__main__':
    main() 