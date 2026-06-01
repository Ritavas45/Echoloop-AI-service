import argparse
import os
import random
import shutil
from pathlib import Path

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
CLASS_NAMES = ['Reuse', 'Refurbish', 'Repair', 'Recycle']

# Map archive folders to the four model output classes.
# The archive sources are grouped by visual condition.
SOURCE_CLASS_MAPPING = {
    'archive/Image_brokenphones': 'Repair',
    'archive/Image_phones': 'Recycle',
    'archive-2/good': 'Reuse',
    'archive-2/oil': 'Refurbish',
    'archive-2/stain': 'Refurbish',
    'archive-2/scratch': 'Repair'
}

SKIP_FOLDERS = {'archive-2/ground_truth_1', 'archive-2/ground_truth_2'}


def is_image_file(path: Path) -> bool:
    return path.suffix in VALID_EXTENSIONS


def collect_images(source_dir: Path):
    images = []
    if not source_dir.exists():
        return images

    # Collect all image files in the directory only (do not descend into annotation/mask dirs).
    for file_path in sorted(source_dir.iterdir()):
        if file_path.is_file() and is_image_file(file_path):
            images.append(file_path)
    return images


def build_dataset(output_root: Path, train_fraction: float = 0.8, seed: int = 42, force: bool = False):
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output root {output_root} already exists and is not empty. "
                "Use --force to recreate it."
            )
        shutil.rmtree(output_root)

    train_root = output_root / 'train'
    val_root = output_root / 'val'

    print(f"Preparing dataset in {output_root}")
    print(f"Train fraction: {train_fraction}")
    output_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    samples_by_class = {cls: [] for cls in CLASS_NAMES}

    for rel_src, class_name in SOURCE_CLASS_MAPPING.items():
        source_dir = Path(rel_src)
        if source_dir.resolve().samefile(Path('archive-2/ground_truth_1')) or source_dir.resolve().samefile(Path('archive-2/ground_truth_2')):
            continue

        if class_name not in CLASS_NAMES:
            raise ValueError(f"Mapped class '{class_name}' is not a supported class: {CLASS_NAMES}")

        if not source_dir.exists():
            print(f"Warning: source directory does not exist: {source_dir}")
            continue

        images = collect_images(source_dir)
        if not images:
            print(f"No image files found in {source_dir}")
            continue

        for img_path in images:
            samples_by_class[class_name].append((class_name, img_path))

    total_samples = sum(len(v) for v in samples_by_class.values())
    print(f"Found {total_samples} archive image samples across {len(samples_by_class)} classes.")

    random.seed(seed)

    for class_name, sample_entries in samples_by_class.items():
        if not sample_entries:
            print(f"Skipping empty class: {class_name}")
            continue

        random.shuffle(sample_entries)
        split_index = int(len(sample_entries) * train_fraction)
        train_entries = sample_entries[:split_index]
        val_entries = sample_entries[split_index:]

        print(f"Class {class_name}: {len(train_entries)} train, {len(val_entries)} val")

        write_samples_for_split(train_root, class_name, train_entries)
        write_samples_for_split(val_root, class_name, val_entries)

    print("Dataset preparation finished.")
    print(f"Train root: {train_root}")
    print(f"Val root: {val_root}")


def write_samples_for_split(split_root: Path, class_name: str, samples):
    class_root = split_root / class_name
    class_root.mkdir(parents=True, exist_ok=True)

    for index, (_, img_path) in enumerate(samples, start=1):
        sample_name = f"sample_{index:05d}_{img_path.stem}"
        sample_dir = class_root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        output_image = sample_dir / f"view_1{img_path.suffix.lower()}"
        shutil.copy2(img_path, output_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare archive images into train/val folders grouped by class.')
    parser.add_argument('--output-root', default='data/archive_prepared', help='Output root for train/val folders.')
    parser.add_argument('--train-fraction', type=float, default=0.8, help='Fraction of samples to place into train.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/val split.')
    parser.add_argument('--force', action='store_true', help='Remove existing output root and recreate dataset.')
    args = parser.parse_args()

    build_dataset(Path(args.output_root), train_fraction=args.train_fraction, seed=args.seed, force=args.force)
