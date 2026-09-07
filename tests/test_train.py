import pytest

torch = pytest.importorskip("torch")

from dlquanttime.evaluate import build_argparser as eval_argparser  # noqa: E402
from dlquanttime.evaluate import evaluate  # noqa: E402
from dlquanttime.train import build_argparser, load_checkpoint, train  # noqa: E402


def _tiny_train_args(model, tmp_path, checkpoint=False):
    parser = build_argparser()
    argv = [
        "--model",
        model,
        "--n-train",
        "12",
        "--n-val",
        "4",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--ar-order",
        "2",
    ]
    if checkpoint:
        argv += ["--checkpoint", str(tmp_path / "model.pt")]
    return parser.parse_args(argv)


@pytest.mark.parametrize(
    "model_name", ["dc_densenet", "mc_densenet_ggg", "conv_autoencoder", "sequential_cnn"]
)
def test_train_runs_one_epoch_for_all_models(model_name, tmp_path):
    args = _tiny_train_args(model_name, tmp_path)
    model = train(args)
    assert isinstance(model, torch.nn.Module)


def test_train_saves_and_evaluate_loads_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    train_args = _tiny_train_args("dc_densenet", tmp_path, checkpoint=True)
    train(train_args)
    assert checkpoint_path.exists()

    loaded_model = load_checkpoint(str(checkpoint_path))
    assert isinstance(loaded_model, torch.nn.Module)

    eval_args = eval_argparser().parse_args(
        [
            "--model",
            "dc_densenet",
            "--n-test",
            "4",
            "--ar-order",
            "2",
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    results = evaluate(eval_args)
    assert set(results.keys())
