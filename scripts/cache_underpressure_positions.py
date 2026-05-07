import argparse
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache FK joint positions into UnderPressure preprocessed .pth files."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to UnderPressure/dataset containing S*/preprocessed/*.pth.",
    )
    parser.add_argument(
        "--underpressure-repo",
        required=True,
        help="Path to the official UnderPressure repo containing anim.py and data.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute positions even if a file already has a positions key.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak copy beside each .pth before modifying it.",
    )
    return parser.parse_args()


def load_underpressure_fk(repo_path):
    repo_path = Path(repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    import anim
    from data import TOPOLOGY

    return anim, TOPOLOGY


def cache_positions(path, anim, topology, overwrite=False, backup=False):
    item = torch.load(path, map_location="cpu", weights_only=False)
    if "positions" in item and not overwrite:
        return "skipped"

    required = {"angles", "skeleton", "trajectory"}
    missing = required.difference(item.keys())
    if missing:
        return f"missing:{','.join(sorted(missing))}"

    positions = anim.FK(item["angles"], item["skeleton"], item["trajectory"], topology)
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu()
    else:
        positions = torch.as_tensor(positions).cpu()

    item["positions"] = positions

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    torch.save(item, path)
    return "cached"


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    files = sorted(dataset_root.glob("S*/preprocessed/*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth files found under {dataset_root}/S*/preprocessed")

    anim, topology = load_underpressure_fk(args.underpressure_repo)
    counts = {}
    for path in tqdm(files, desc="Caching positions"):
        status = cache_positions(
            path,
            anim,
            topology,
            overwrite=args.overwrite,
            backup=args.backup,
        )
        counts[status] = counts.get(status, 0) + 1

    print("Done.")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
