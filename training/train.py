#!/usr/bin/env python3
"""
S2F training script.
Usage:
  python -m training.train --data path/to/dataset --model single_cell --epochs 100
  python -m training.train --data path/to/dataset --model spheroid --epochs 50
"""
import os
import sys
import argparse

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)


def main():
    parser = argparse.ArgumentParser(description='Train S2F model')
    parser.add_argument('--data', required=True, help='Path to dataset (must have train/ and test/ subfolders)')
    parser.add_argument('--model', choices=['single_cell', 'spheroid'], default='single_cell',
                        help='Model type: single_cell (with substrate) or spheroid')
    parser.add_argument('--substrate', default=None,
                        help='Substrate name for single_cell when metadata not in dataset (e.g. fibroblasts_PDMS)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--save_dir', default='ckp', help='Checkpoint save directory')
    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_augment', action='store_true', help='Disable augmentations')
    parser.add_argument('--use_force_consistency', action='store_true')
    parser.add_argument('--force_target', choices=['mean', 'sum'], default='mean')
    args = parser.parse_args()

    from data.cell_dataset import prepare_data
    from models.s2f_model import create_s2f_model
    from training.s2f_trainer import train_s2f

    use_settings = args.model == 'single_cell'
    substrate = args.substrate or 'fibroblasts_PDMS'
    return_metadata = use_settings

    print(f"Loading data from {args.data} (model={args.model})")
    train_loader, val_loader = prepare_data(
        args.data,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        use_augmentations=not args.no_augment,
        train_test_sep_folder=True,
        return_metadata=return_metadata,
        substrate=substrate if use_settings else None,
    )

    in_channels = 3 if use_settings else 1
    generator, discriminator = create_s2f_model(in_channels=in_channels)

    if args.resume:
        ckpt = __import__('torch').load(args.resume, map_location='cpu', weights_only=False)
        generator.load_state_dict(ckpt.get('generator_state_dict', ckpt), strict=True)
        print(f"Resumed from {args.resume}")

    history = train_s2f(
        generator, discriminator,
        train_loader, val_loader,
        device=args.device,
        num_epochs=args.epochs,
        g_lr=args.g_lr, d_lr=args.d_lr,
        save_dir=args.save_dir,
        loaded_metadata=return_metadata,
        use_settings=use_settings,
        use_force_consistency=args.use_force_consistency,
        force_consistency_target=args.force_target,
    )
    print(f"Training complete. Checkpoints saved to {args.save_dir}")


if __name__ == '__main__':
    main()
