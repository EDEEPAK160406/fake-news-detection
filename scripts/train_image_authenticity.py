from __future__ import annotations

import argparse

from app.services.image_authenticity import image_authenticity_singleton


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the image authenticity detector.")
    parser.add_argument(
        "--data-dir",
        default="data/image_authenticity",
        help="Directory containing ai_generated/ and real/ subfolders.",
    )
    parser.add_argument("--cv", type=int, default=0, help="Run cross-validation training with this many folds (0 to skip)")
    parser.add_argument("--augment", action="store_true", help="Enable simple data augmentation to expand small classes during training")
    parser.add_argument("--min-per-class", type=int, default=200, help="Minimum samples per class after augmentation")
    args = parser.parse_args()

    if args.cv and args.cv > 1:
        accuracy, samples = image_authenticity_singleton.fit_with_cv(args.data_dir, cv=args.cv)
        print(f"Trained image authenticity model with {args.cv}-fold CV on {samples} samples")
        print(f"Mean CV accuracy: {accuracy:.4f}")
    else:
        accuracy, samples = image_authenticity_singleton.fit_from_directories(args.data_dir, augment=args.augment, min_samples_per_class=args.min_per_class)
        print(f"Trained image authenticity model on {samples} samples")
        print(f"Training accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()