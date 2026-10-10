"""Train the baseline ECGModel using only PTB-XL train and validation splits.

The test split is counted for safety but is never converted to tensors, passed
to the model, or used to select a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import ECGModel


# Save an image file without opening a separate plotting window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
REQUIRED_NPZ_KEYS = {"X", "Y", "ecg_id", "patient_id", "strat_fold", "split"}
EXPECTED_SPLIT_COUNTS = {"train": 80, "validation": 10, "test": 10}


def set_seed(seed: int) -> None:
    """Fix the main random sources so repeated baseline runs are comparable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_validate_data(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Load the NPZ file, validate it, and return X/Y plus split labels.

    This function checks all arrays for NaN/inf before learning begins. It does
    not create a test DataLoader or return test tensors.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as archive:
        missing_keys = REQUIRED_NPZ_KEYS.difference(archive.files)
        if missing_keys:
            raise KeyError(f"Dataset is missing required keys: {sorted(missing_keys)}")
        x_data = archive["X"]
        y_data = archive["Y"]
        split_labels = archive["split"].astype(str)

    if x_data.dtype != np.float32 or y_data.dtype != np.float32:
        raise TypeError(f"X/Y must be float32, got X={x_data.dtype}, Y={y_data.dtype}")
    if x_data.ndim != 3 or x_data.shape[1:] != (1000, 1):
        raise ValueError(f"X must have shape [records, 1000, 1], got {x_data.shape}")
    if y_data.ndim != 3 or y_data.shape[1:] != (1000, 11):
        raise ValueError(f"Y must have shape [records, 1000, 11], got {y_data.shape}")
    if len(x_data) != len(y_data) or len(x_data) != len(split_labels):
        raise ValueError("X, Y, and split arrays must have the same record count")
    if not np.isfinite(x_data).all() or not np.isfinite(y_data).all():
        raise ValueError("X or Y contains NaN or infinity; training was stopped before it began")

    split_counts = {name: int(np.sum(split_labels == name)) for name in EXPECTED_SPLIT_COUNTS}
    unknown_splits = sorted(set(split_labels).difference(EXPECTED_SPLIT_COUNTS))
    if split_counts != EXPECTED_SPLIT_COUNTS or unknown_splits:
        raise ValueError(
            f"Unexpected dataset split counts: {split_counts}; unknown labels: {unknown_splits}"
        )
    return x_data, y_data, split_labels, split_counts


def make_loader(
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Convert [records, time, channels] arrays to Conv1d's [records, channels, time]."""
    x_channels_first = np.ascontiguousarray(np.transpose(x_data, (0, 2, 1)))
    y_channels_first = np.ascontiguousarray(np.transpose(y_data, (0, 2, 1)))
    dataset = TensorDataset(torch.from_numpy(x_channels_first), torch.from_numpy(y_channels_first))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Safe and reproducible on Windows for this small baseline dataset.
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def mean_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training epoch when optimizer is set, otherwise run validation."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise RuntimeError(
                    f"Prediction/target shape mismatch: {predictions.shape} vs {targets.shape}"
                )
            loss = criterion(predictions, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_examples = inputs.size(0)
            total_loss += loss.item() * batch_examples
            total_examples += batch_examples

    return total_loss / total_examples


def ensure_outputs_are_safe(paths: tuple[Path, Path, Path], overwrite: bool) -> None:
    """Prevent replacement of previous training artifacts without --overwrite."""
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths and not overwrite:
        joined_paths = "\n".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Existing training outputs were found. Inspect them first, then rerun with --overwrite "
            f"only if replacement is intended:\n{joined_paths}"
        )


def save_history_csv(history: list[dict[str, float | int]], output_path: Path) -> None:
    """Save one loss row per completed epoch."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)


def save_loss_plot(history: list[dict[str, float | int]], output_path: Path) -> None:
    """Save train/validation loss curves after training is complete."""
    epochs = [int(row["epoch"]) for row in history]
    train_losses = [float(row["train_loss"]) for row in history]
    validation_losses = [float(row["validation_loss"]) for row in history]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_losses, marker="o", label="Train loss")
    axis.plot(epochs, validation_losses, marker="o", label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("Baseline ECGModel training history")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def train(arguments: argparse.Namespace) -> None:
    """Run baseline training using train/validation records only."""
    checkpoint_path = Path("checkpoints") / "best_model.pth"
    loss_plot_path = Path("results") / "training_loss.png"
    history_path = Path("results") / "training_history.csv"
    output_paths = (checkpoint_path, loss_plot_path, history_path)
    ensure_outputs_are_safe(output_paths, arguments.overwrite)

    set_seed(arguments.seed)
    x_data, y_data, split_labels, split_counts = load_and_validate_data(arguments.dataset)

    train_mask = split_labels == "train"
    validation_mask = split_labels == "validation"
    # The test mask is deliberately used only to report its already-validated count.
    test_count = split_counts["test"]

    loader_generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = make_loader(
        x_data[train_mask], y_data[train_mask], arguments.batch_size, shuffle=True, generator=loader_generator
    )
    validation_loader = make_loader(
        x_data[validation_mask], y_data[validation_mask], arguments.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)

    print("===== Baseline training configuration =====")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seed: {arguments.seed}")
    print(f"Batch size: {arguments.batch_size}")
    print(f"Maximum epochs: {arguments.epochs}")
    print(f"Learning rate: {arguments.learning_rate}")
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(
        "Records — "
        f"train: {len(train_loader.dataset)}, validation: {len(validation_loader.dataset)}, "
        f"test withheld: {test_count}"
    )
    print("Test records are not passed to a DataLoader, model, loss, or checkpoint decision.")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, arguments.epochs + 1):
        train_loss = mean_loss(model, train_loader, criterion, device, optimizer=optimizer)
        validation_loss = mean_loss(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:02d}/{arguments.epochs} | "
            f"Train Loss: {train_loss:.6f} | Validation Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_validation_loss,
                    "seed": arguments.seed,
                    "batch_size": arguments.batch_size,
                    "learning_rate": arguments.learning_rate,
                },
                checkpoint_path,
            )
            print(f"  Saved new best checkpoint: epoch={best_epoch}, validation_loss={best_validation_loss:.6f}")

    save_history_csv(history, history_path)
    save_loss_plot(history, loss_plot_path)
    print("\n===== Training complete =====")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Loss graph: {loss_plot_path.resolve()}")
    print(f"History CSV: {history_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline 1-lead to 12-lead ECGModel.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data") / "processed" / "ptbxl_100hz_100records.npz",
        help="Prepared NPZ dataset path.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED}).")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Adam learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS, help=f"Maximum epochs, 1-{MAX_EPOCHS} (default: {MAX_EPOCHS})."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing checkpoint and result files.",
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.epochs <= MAX_EPOCHS:
        parser.error(f"--epochs must be between 1 and {MAX_EPOCHS}")
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    train(arguments)


if __name__ == "__main__":
    main()
