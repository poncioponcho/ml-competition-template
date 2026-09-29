"""Regression guard for the learning-rate a *resumed* run actually trains with.

The bug this file exists for: ``optimizer.load_state_dict()`` restores the whole
param group, including ``lr``. A cosine schedule that has run to the end of its
budget stores ``lr = 0``, so a resumed run trains its entire remaining budget at
lr = 0. Nothing about the log looks wrong - every step prints, the loss wanders
with batch noise - but the saved weights come out **bit-identical** to the
checkpoint the run started from.

That is not hypothetical: fold 0 was resumed from epoch 12 to epoch 24 on
2026-09-28 and burned ~2.9 h producing a checkpoint whose weights hashed the same
as the one it loaded. It was only caught by hashing both checkpoints.

The scheduling helper lives in a script rather than a package module, so it is
imported by path.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Advancing a schedule without stepping the optimizer is the whole point of
# these tests, so torch's ordering warning is expected noise here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*lr_scheduler.step.*before.*optimizer.step.*:UserWarning"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_segmentation_under_test",
        REPO_ROOT / "scripts" / "train_segmentation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_LR = 0.005
EPOCHS = 24
RESUME_AT = 12


def _finished_cosine_optimizer(param_count: int = 2):
    """An optimizer whose lr has annealed to exactly 0, like a saved checkpoint."""
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(param_count)]
    optimizer = torch.optim.SGD(params, lr=BASE_LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    for _ in range(EPOCHS):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0, "precondition: annealed to zero"
    return optimizer


def _closed_form(epoch: int, total: int, base_lr: float = BASE_LR) -> float:
    return base_lr * (1 + math.cos(math.pi * epoch / total)) / 2


def test_a_finished_cosine_really_does_store_lr_zero():
    """Documents the trap: this is what a checkpoint at the end of a run holds."""
    assert _finished_cosine_optimizer().param_groups[0]["lr"] == 0.0


def test_resumed_run_does_not_train_at_zero_lr():
    """The regression: resuming from an lr=0 checkpoint must restore a live lr."""
    train = _load_train_module()
    optimizer = _finished_cosine_optimizer()
    train.make_scheduler(torch, optimizer, EPOCHS, RESUME_AT, BASE_LR)
    assert optimizer.param_groups[0]["lr"] > 0.0


@pytest.mark.parametrize("start_epoch", [0, 1, 5, RESUME_AT, 23])
def test_resumed_schedule_lands_on_the_uninterrupted_curve(start_epoch: int):
    """Fast-forwarding must reproduce the lr the run would have had anyway."""
    train = _load_train_module()
    optimizer = _finished_cosine_optimizer()
    scheduler = train.make_scheduler(torch, optimizer, EPOCHS, start_epoch, BASE_LR)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        _closed_form(start_epoch, EPOCHS), rel=1e-12
    )
    # And it keeps tracking the same curve on the epochs that follow.
    for offset in (1, 2, 3):
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            _closed_form(start_epoch + offset, EPOCHS), rel=1e-12
        )


def test_fresh_run_still_starts_at_the_configured_lr():
    """A run with no checkpoint must be unaffected by the resume fix."""
    train = _load_train_module()
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    optimizer = torch.optim.SGD(params, lr=BASE_LR, momentum=0.9)
    train.make_scheduler(torch, optimizer, EPOCHS, 0, BASE_LR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR, rel=1e-12)
