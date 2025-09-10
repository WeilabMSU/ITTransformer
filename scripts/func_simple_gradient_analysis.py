#!/usr/bin/env python3
"""
Simple Gradient + Token-Importance (feature-dim–fair) — Minimal Outputs

What this script does:
- Loads a fine-tuned ITT model and a single feature file.
- Computes input gradients and Grad×Input saliency per token for part0/1/2 and atom_fea.
- Masks padded atoms.
- Produces TWO per-token scores:
    1) RAW  : sum_f |x_f * g_f|
    2) FAIR : mean_f |x_f * g_f|  (feature-dimension–normalized; best for cross-type comparisons)
- Aggregates FAIR scores into: counts, totals, shares, mean per token, Gini, head Top-5%/Top-10%.
- Saves:
    * token_importance_type_summary.csv  (ALWAYS; one row for this sample, prefix headers)
    * combined_results.npz  (OPTIONAL via --save-npz) containing:
        - gradients per stream
        - per-feature saliency per stream (abs(input * grad))
        - per-token FAIR & RAW arrays
        - atom nonpad_mask

Usage:
python simple_gradient_analysis.py \
  --model-path /path/to/checkpoint.pt \
  --feature-file /path/to/single_file.npz \
  --output-path ./out \
  --device cuda \
  [--standardize-features --feature-scaler /path/to/scaler.joblib] \
  [--save-npz]
"""

import os
import argparse
import csv
import numpy as np
import torch


# Project-specific imports (keep consistent with your codebase)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.itt.utils import IndividualFeatureDataset, collate_fn
from main.itt.configuration_itt import IttConfig
from main.itt.modeling_itt_cross import IttCrossSequenceClassifier


def parse_args():
    p = argparse.ArgumentParser(description='IT-Transformer Gradients + Token-Importance (minimal outputs)')
    p.add_argument('--model-path', type=str, required=True, help='Path to trained finetuned model checkpoint')
    p.add_argument('--feature-file', type=str, required=True, help='Path to single .npz feature file')
    p.add_argument('--output-path', type=str, required=True, help='Folder to save outputs')
    p.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    p.add_argument('--standardize-features', action='store_true', help='Standardize features (tabular parts) with a scaler')
    p.add_argument('--feature-scaler', type=str, default=None, help='Path to feature scaler (joblib) if standardizing')
    p.add_argument('--save-npz', action='store_true', help='If set, save combined_results.npz with gradients, saliency, and per-token arrays')
    p.add_argument('--save-specific-csv', action='store_true', help='If set, save .csv by the input feature-file name')
    return p.parse_args()


class SimpleGradientAnalyzer:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'config' in checkpoint:
            cfg = checkpoint['config']
            if isinstance(cfg, dict):
                self.config = IttConfig(**cfg)
            else:
                cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg.__dict__)
                self.config = IttConfig(**cfg_dict)
        else:
            self.config = IttConfig()

        self.model = IttCrossSequenceClassifier(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        print(f"✓ Model loaded on {self.device}")

    def analyze_single_input(self, feature_file: str, feature_scaler=None):
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        # ---- Single-sample dataset/loader
        file_id = os.path.splitext(os.path.basename(feature_file))[0]
        indices = np.array([0]); file_ids = [file_id]; values = np.array([0.0])
        dataset = IndividualFeatureDataset((indices, file_ids, values), os.path.dirname(feature_file), feature_scaler)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        (inputs, _labels) = next(iter(loader))

        # ---- Unpack inputs
        part0, part1, part2, graph_data_tuple = inputs
        part0 = part0.to(torch.float32).to(self.device).requires_grad_(True)  # [B, 1, 750]
        part1 = part1.to(torch.float32).to(self.device).requires_grad_(True)  # [B, 7, 750]
        part2 = part2.to(torch.float32).to(self.device).requires_grad_(True)  # [B, 42, 500]

        atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx = graph_data_tuple
        atom_fea = atom_fea.to(torch.float32).to(self.device).requires_grad_(True)  # [B, A, 128]
        nbr_fea = nbr_fea.to(self.device)
        nbr_fea_idx = nbr_fea_idx.to(self.device)
        cluster_indices = cluster_indices.to(self.device)
        crystal_atom_idx = [idx.to(self.device) for idx in crystal_atom_idx]
        graph_data = (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices, crystal_atom_idx)

        # ---- Capture padded_atom_fea act & grad via hooks
        act, grads = {}, {}

        def fwd_hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            act["padded_atom_fea"] = t  # [B, Amax, 128]

        def bwd_hook(_m, _gin, gout):
            g = gout[0] if isinstance(gout, (tuple, list)) else gout
            grads["padded_atom_fea"] = g.detach()

        target_mod = self.model.itt_cross.graph_cluster_encoder
        h_fwd = target_mod.register_forward_hook(fwd_hook)
        h_bwd = target_mod.register_full_backward_hook(bwd_hook)

        # ---- Forward & target selection
        outputs = self.model(
            input_part0=part0, input_part1=part1, input_part2=part2,
            graph_data=graph_data, use_cross_attention_mask=False
        )
        h_fwd.remove()
        predictions = outputs[0]  # [B, C] or [B, 1]
        if predictions.shape[1] > 1:
            cls = predictions.argmax(dim=1)
            y = predictions[0, cls[0]]
            print(f"Target: class={cls.item()}, logit={y.item():.6f}")
        else:
            y = predictions.view(-1)[0]
            print(f"Target: {y.item():.6f}")

        # ---- Backward
        self.model.zero_grad(set_to_none=True)
        for t in (part0, part1, part2, atom_fea):
            if t.grad is not None:
                t.grad.zero_()
        y.backward()
        h_bwd.remove()

        pad_act = act.get('padded_atom_fea', None)          # [B, Amax, 128]
        pad_grad = grads.get('padded_atom_fea', None)        # [B, Amax, 128]

        # ---- Gradients (fallback zeros if missing)
        grad_part0 = part0.grad.clone().detach() if part0.grad is not None else torch.zeros_like(part0)
        grad_part1 = part1.grad.clone().detach() if part1.grad is not None else torch.zeros_like(part1)
        grad_part2 = part2.grad.clone().detach() if part2.grad is not None else torch.zeros_like(part2)
        if pad_grad is not None:
            grad_atom = pad_grad
        else:
            if pad_act is not None:
                grad_atom = torch.zeros_like(pad_act)
            else:
                Amax = getattr(self.config, 'max_atom_number', 256)
                d_atom = getattr(self.config, 'd3', 128)
                grad_atom = torch.zeros(1, Amax, d_atom, device=self.device)

        # ---- Per-feature saliency = |input * grad|
        sal_part0 = (part0 * grad_part0).abs()
        sal_part1 = (part1 * grad_part1).abs()
        sal_part2 = (part2 * grad_part2).abs()

        if pad_act is None:
            Amax = getattr(self.config, 'max_atom_number', 256)
            sal_atom = torch.zeros(1, Amax, getattr(self.config, 'd3', 128), device=self.device)
            nonpad_mask = torch.zeros(1, Amax, dtype=torch.bool, device=self.device)
        else:
            sal_atom = (pad_act * grad_atom).abs()
            with torch.no_grad():
                nonpad_mask = (pad_act.abs().sum(dim=-1) > 0)  # [B, Amax]

        # ---- Token scores: RAW (sum_f) and FAIR (mean_f)
        def token_scores_pair_from_sal(sal: torch.Tensor):
            if sal.ndim == 3:
                raw = sal.sum(dim=-1)    # [B, T]
                fair = sal.mean(dim=-1)  # [B, T]
            elif sal.ndim == 2:
                raw = sal.sum(dim=-1, keepdim=True)    # [B, 1]
                fair = sal.mean(dim=-1, keepdim=True)  # [B, 1]
            else:
                sal2 = sal.view(sal.size(0), -1)
                raw = sal2.sum(dim=-1, keepdim=True)
                fair = sal2.mean(dim=-1, keepdim=True)
            return raw, fair

        p0_raw, p0_fair = token_scores_pair_from_sal(sal_part0)
        p1_raw, p1_fair = token_scores_pair_from_sal(sal_part1)
        p2_raw, p2_fair = token_scores_pair_from_sal(sal_part2)
        a_raw_full, a_fair_full = token_scores_pair_from_sal(sal_atom)
        pA_raw = a_raw_full * nonpad_mask
        pA_fair = a_fair_full * nonpad_mask

        # ---- Convert to numpy (B=1 squeeze)
        p0_raw = p0_raw.squeeze(0).detach().cpu().numpy()   # len=1
        p1_raw = p1_raw.squeeze(0).detach().cpu().numpy()   # len=7
        p2_raw = p2_raw.squeeze(0).detach().cpu().numpy()   # len=42
        pA_raw = pA_raw.squeeze(0).detach().cpu().numpy()   # len<=256
        p0 = p0_fair.squeeze(0).detach().cpu().numpy()
        p1 = p1_fair.squeeze(0).detach().cpu().numpy()
        p2 = p2_fair.squeeze(0).detach().cpu().numpy()
        pa = pA_fair.squeeze(0).detach().cpu().numpy()

        # ---- Aggregates (FAIR arrays)
        def _count_present(arr: np.ndarray) -> int:
            return int((arr > 0).sum())  # for atoms, padded positions are zeroed

        counts = {"part0": max(1, p0.size), "part1": max(1, p1.size),
                  "part2": max(1, p2.size), "atom_fea": _count_present(pa)}

        totals = {"part0": float(p0.sum()), "part1": float(p1.sum()),
                  "part2": float(p2.sum()), "atom_fea": float(pa.sum())}
        grand_total = sum(totals.values()) + 1e-12
        shares = {k: float(v / grand_total) for k, v in totals.items()}

        raw_totals = {"part0": float(p0_raw.sum()), "part1": float(p1_raw.sum()),
                      "part2": float(p2_raw.sum()), "atom_fea": float(pA_raw.sum())}
        raw_grand_total = sum(raw_totals.values()) + 1e-12
        raw_shares = {k: float(v / raw_grand_total) for k, v in raw_totals.items()}

        means = {k: float(totals[k] / max(counts[k], 1)) for k in ["part0", "part1", "part2", "atom_fea"]}

        def _head_share(arr: np.ndarray, q: float) -> float:
            if arr.size == 0 or np.all(arr == 0): return 0.0
            k = max(1, int(np.ceil(q * arr.size)))
            idx = np.argpartition(-arr, k - 1)[:k]
            return float(arr[idx].sum() / (arr.sum() + 1e-12))

        def _gini(arr: np.ndarray) -> float:
            a = np.asarray(arr, dtype=np.float64)
            a = a[a >= 0]
            if a.size == 0 or a.sum() == 0: return 0.0
            a.sort()
            n = a.size
            cum = np.cumsum(a)
            return float((n + 1 - 2 * (cum.sum() / a.sum()) / n))

        gini = { "part0": _gini(p0), "part1": _gini(p1), "part2": _gini(p2), "atom_fea": _gini(pa) }
        head5 = { k: _head_share(arr, 0.05) for k, arr in zip(["part0","part1","part2","atom_fea"], [p0,p1,p2,pa]) }
        head10 = { k: _head_share(arr, 0.10) for k, arr in zip(["part0","part1","part2","atom_fea"], [p0,p1,p2,pa]) }

        # ---- Package numpy for optional NPZ save
        npz_payload = {
            # Gradients (float32)
            "grad_part0": grad_part0.detach().cpu().numpy(),
            "grad_part1": grad_part1.detach().cpu().numpy(),
            "grad_part2": grad_part2.detach().cpu().numpy(),
            "grad_padded_atom_fea": grad_atom.detach().cpu().numpy(),

            # Per-feature saliency = |input * grad|
            "saliency_part0": sal_part0.detach().cpu().numpy(),
            "saliency_part1": sal_part1.detach().cpu().numpy(),
            "saliency_part2": sal_part2.detach().cpu().numpy(),
            "saliency_padded_atom_fea": sal_atom.detach().cpu().numpy(),

            # Per-token FAIR (feature-dim mean) and RAW (feature-dim sum)
            "token_fair_part0": p0, "token_fair_part1": p1, "token_fair_part2": p2, "token_fair_atom_fea": pa,
            "token_raw_part0": p0_raw, "token_raw_part1": p1_raw, "token_raw_part2": p2_raw, "token_raw_atom_fea": pA_raw,

            # Atom mask (so you can re-mask later)
            "atom_nonpad_mask": nonpad_mask.squeeze(0).detach().cpu().numpy().astype(np.bool_),

            # Save feture and model information
            "feature_file": feature_file,
            "model_path": self.model_path,
        }

        # ---- Type summary (dict of scalar metrics)
        summary = {
            # counts
            "part0_count": counts["part0"], "part1_count": counts["part1"],
            "part2_count": counts["part2"], "atom_fea_count": counts["atom_fea"],

            # FAIR aggregates
            "part0_total_fair": totals["part0"], "part1_total_fair": totals["part1"],
            "part2_total_fair": totals["part2"], "atom_fea_total_fair": totals["atom_fea"],
            "part0_share_fair": shares["part0"], "part1_share_fair": shares["part1"],
            "part2_share_fair": shares["part2"], "atom_fea_share_fair": shares["atom_fea"],
            "part0_mean_per_token_fair": means["part0"], "part1_mean_per_token_fair": means["part1"],
            "part2_mean_per_token_fair": means["part2"], "atom_fea_mean_per_token_fair": means["atom_fea"],

            # RAW aggregates (reference)
            "part0_total_raw": raw_totals["part0"], "part1_total_raw": raw_totals["part1"],
            "part2_total_raw": raw_totals["part2"], "atom_fea_total_raw": raw_totals["atom_fea"],
            "part0_share_raw": raw_shares["part0"], "part1_share_raw": raw_shares["part1"],
            "part2_share_raw": raw_shares["part2"], "atom_fea_share_raw": raw_shares["atom_fea"],

            # Concentration on FAIR arrays
            "part0_gini_fair": gini["part0"], "part1_gini_fair": gini["part1"],
            "part2_gini_fair": gini["part2"], "atom_fea_gini_fair": gini["atom_fea"],
            "part0_head_top5pct_fair": head5["part0"], "part1_head_top5pct_fair": head5["part1"],
            "part2_head_top5pct_fair": head5["part2"], "atom_fea_head_top5pct_fair": head5["atom_fea"],
            "part0_head_top10pct_fair": head10["part0"], "part1_head_top10pct_fair": head10["part1"],
            "part2_head_top10pct_fair": head10["part2"], "atom_fea_head_top10pct_fair": head10["atom_fea"],
        }

        return npz_payload, summary


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Optional scaler
    feature_scaler = None
    if args.standardize_features:
        if not args.feature_scaler or not os.path.exists(args.feature_scaler):
            raise ValueError('--feature-scaler file required when --standardize-features is set')
        from joblib import load
        feature_scaler = load(args.feature_scaler)
        print(f'✓ Loaded feature scaler from {args.feature_scaler}')

    print("\n==================== START ====================")
    npz_payload, summary = SimpleGradientAnalyzer(args.model_path, args.device).analyze_single_input(
        args.feature_file, feature_scaler
    )
    print("===================== DONE ====================\n")

    # ---- ALWAYS save compact one-row CSV with prefixed headers
    if args.save_specific_csv:
        base_name = os.path.splitext(os.path.basename(args.feature_file))[0]           # e.g., "example"
        type_csv = os.path.join(args.output_path, f"{base_name}.csv")
    else:
        type_csv = os.path.join(args.output_path, "token_importance_type_summary.csv")
    with open(type_csv, "w", newline="") as f:
        w = csv.writer(f)
        headers = list(summary.keys())
        w.writerow(headers)
        w.writerow([summary[h] for h in headers])
    print(f"✓ Type summary saved: {type_csv}")

    # ---- OPTIONALLY save a combined NPZ with everything needed to recompute analyses
    if args.save_npz:
        npz_path = os.path.join(args.output_path, "combined_results.npz")
        np.savez(npz_path, **npz_payload)
        print(f"✓ Combined NPZ saved: {npz_path}")
    else:
        print("Skipped NPZ (run with --save-npz to save gradients + saliency + per-token arrays).")

    # ---- Console view (brief)
    print("\n===== TOKEN-TYPE COMPARISON (FAIR) =====")
    for k in ("part0", "part1", "part2", "atom_fea"):
        tot = summary[f"{k}_total_fair"]; shr = summary[f"{k}_share_fair"] * 100
        mean = summary[f"{k}_mean_per_token_fair"]; cnt = summary[f"{k}_count"]
        print(f"{k:>10s}: total={tot:.6g} (share={shr:5.2f}%), mean/token={mean:.6g} over n={cnt}")
    print("=========================================")


if __name__ == '__main__':
    main()
