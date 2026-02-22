#!/usr/bin/env python3
"""
S2F evaluation script.
Metrics: MSE, MS-SSIM, Pixel Correlation, Relative Magnitude Error, Force Sum/Mean correlation.

Usage:
  python -m training.evaluate --model single_cell --checkpoint ckp/best_checkpoint.pth --data path/to/test
  python -m training.evaluate --model spheroid --checkpoint ckp/best_checkpoint.pth --data path/to/test
"""
import os
import sys
import argparse

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)


def main():
    parser = argparse.ArgumentParser(description='Evaluate S2F model')
    parser.add_argument('--data', required=True, help='Path to test folder (subfolders with BF_001.tif, *_gray.jpg)')
    parser.add_argument('--model', choices=['single_cell', 'spheroid'], default='single_cell')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--substrate', default='fibroblasts_PDMS', help='Substrate for single_cell')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=0.0, help='Threshold for heatmap metrics')
    parser.add_argument('--output', default=None, help='Optional CSV path for per-sample metrics')
    parser.add_argument('--save_plots', default=None, help='Directory to save prediction plots')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    from data.cell_dataset import load_folder_data
    from models.s2f_model import create_s2f_model, compute_settings_normalization
    from utils.metrics import (
        evaluate_metrics_on_dataset,
        print_metrics_report,
        gen_prediction_plots,
        detect_tanh_output_model,
    )
    import torch
    import pandas as pd

    use_settings = args.model == 'single_cell'
    config_path = os.path.join(S2F_ROOT, 'config', 'substrate_settings.json')

    print(f"Loading data from {args.data}")
    val_loader = load_folder_data(
        args.data,
        substrate=args.substrate if use_settings else None,
        img_size=args.img_size,
        batch_size=args.batch_size,
        return_metadata=use_settings,
    )

    in_channels = 3 if use_settings else 1
    generator, _ = create_s2f_model(in_channels=in_channels)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    generator.load_state_dict(ckpt.get('generator_state_dict', ckpt), strict=True)

    norm_params = compute_settings_normalization(config_path=config_path) if use_settings else None
    uses_tanh = detect_tanh_output_model(generator)

    results = evaluate_metrics_on_dataset(
        generator,
        val_loader,
        device=args.device,
        description="Evaluating",
        save_predictions=(args.save_plots is not None or args.output is not None),
        threshold=args.threshold,
        use_settings=use_settings,
        normalization_params=norm_params,
        config_path=config_path,
        substrate_override=args.substrate,
    )

    report = {'validation': results}
    print_metrics_report(report, threshold=args.threshold, uses_tanh=uses_tanh)
    print(f"Samples: {len(val_loader.dataset)}")

    if args.save_plots and 'individual_predictions' in results:
        gen_prediction_plots(
            results['individual_predictions'],
            args.save_plots,
            sort_by='mse',
            sort_order='asc',
            threshold=args.threshold,
        )
        print(f"Saved prediction plots to {args.save_plots}")

    if args.output:
        preds = results.get('individual_predictions', [])
        if preds:
            df = pd.DataFrame([{
                'mse': p['mse'],
                'ms_ssim': p['ms_ssim'],
                'pixel_correlation': p['pixel_correlation'],
                'relative_magnitude_error': p.get('wfm_relative_magnitude_error'),
                'force_sum_gt': p['force_sum_gt'],
                'force_sum_pred': p['force_sum_pred'],
            } for p in preds])
            df.to_csv(args.output, index=False)
            print(f"Saved per-sample metrics to {args.output}")
        else:
            # Fallback: write summary only
            with open(args.output.replace('.csv', '_summary.txt'), 'w') as f:
                f.write(f"MSE: {results['heatmap']['mse']:.6f}\n")
                f.write(f"MS-SSIM: {results['heatmap']['ms_ssim']:.4f}\n")
                f.write(f"Pixel Corr: {results['heatmap']['pixel_correlation']:.4f}\n")
                f.write(f"Rel Mag Error: {results['wfm']['relative_magnitude_error']:.4f}\n")
            print(f"Saved summary to {args.output.replace('.csv', '_summary.txt')}")


if __name__ == '__main__':
    main()
