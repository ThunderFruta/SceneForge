from __future__ import annotations

from pathlib import Path


def test_train_cli_batch_argument_is_numeric() -> None:
    from run import build_parser

    args = build_parser().parse_args(
        [
            "train-rgbd-yolo",
            "--data",
            "data.yaml",
            "--output",
            "model.pt",
            "--batch",
            "16",
        ]
    )

    assert args.batch == 16
    assert isinstance(args.batch, int)


def test_train_cli_defaults_to_batch_8_and_patience_5() -> None:
    from run import build_parser

    args = build_parser().parse_args(
        [
            "train-rgbd-yolo",
            "--data",
            "data.yaml",
            "--output",
            "model.pt",
        ]
    )

    assert args.batch == 8
    assert args.patience == 5
    assert args.lr0 is None


def test_train_cli_accepts_lr0() -> None:
    from run import build_parser

    args = build_parser().parse_args(
        [
            "train-rgbd-yolo",
            "--data",
            "data.yaml",
            "--output",
            "model.pt",
            "--lr0",
            "0.001",
        ]
    )

    assert args.lr0 == 0.001


def test_generate_rgbd_dataset_defaults_to_70_20_10_split() -> None:
    from run import build_parser

    args = build_parser().parse_args(
        [
            "generate-rgbd-dataset",
            "--curriculum-stage",
            "1",
        ]
    )

    assert args.train_split == 0.70
    assert args.val_split == 0.20
    assert args.shards == "auto"


def test_generate_rgbd_dataset_accepts_manual_shards() -> None:
    from run import build_parser

    args = build_parser().parse_args(
        [
            "generate-rgbd-dataset",
            "--curriculum-stage",
            "1",
            "--shards",
            "8",
        ]
    )

    assert args.shards == 8


def test_generate_target_rgbd_dataset_can_make_eval_only_split() -> None:
    from run import build_parser

    args = build_parser().parse_args(
        [
            "generate-target-rgbd-dataset",
            "--reference-blend",
            "Assets/Samples/shapes.blend",
            "--output",
            "Datasets/TargetEval/shapes_blend",
            "--eval-only",
            "--exact-first",
            "--shards",
            "4",
        ]
    )

    assert args.eval_only is True
    assert args.exact_first is True
    assert args.shards == 4
    assert args.train_split == 0.70
    assert args.val_split == 0.20


def test_eval_rgbd_yolo_cli_defaults_to_test_split() -> None:
    from run import build_parser

    args = build_parser().parse_args(
        [
            "eval-rgbd-yolo",
            "--data",
            "data_rgbd.yaml",
            "--weights",
            "model.pt",
            "--output",
            "runs/eval",
        ]
    )

    assert args.split == "test"
    assert args.batch == 8


def test_auto_shards_scales_with_dataset_size() -> None:
    from run import _resolve_auto_shards

    assert _resolve_auto_shards("auto", 1) == 1
    assert 1 <= _resolve_auto_shards("auto", 100) <= 32
    assert 1 <= _resolve_auto_shards("auto", 1000) <= 32


def test_yolo26l_rgbd_config_instantiates_four_channel_model() -> None:
    from ultralytics.nn.tasks import SegmentationModel

    root = Path(__file__).resolve().parents[2]
    model = SegmentationModel(str(root / "Configs/YOLO/yolo26l_seg_rgbd.yaml"), ch=4, nc=5, verbose=False)
    first_conv = next(module for module in model.modules() if module.__class__.__name__ == "Conv")

    assert first_conv.conv.in_channels == 4
