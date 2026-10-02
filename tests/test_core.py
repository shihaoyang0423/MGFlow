import numpy as np
import pytest
import torch

from mgflow import Branch, GaussianMixture, MomentState
from mgflow.checkpoint import load_checkpoint, save_checkpoint
from mgflow.distributions.assignment import balanced_assignment
from mgflow.distributions.moments import moments
from mgflow.distributions.objectives import kl_velocity, regression_loss


def reference(k=4, d=5):
    generator = torch.Generator().manual_seed(0)
    mean = torch.randn(k, d, generator=generator, dtype=torch.float64)
    matrix = torch.randn(k, d, d, generator=generator, dtype=torch.float64)
    covariance = matrix @ matrix.transpose(-1, -2) + torch.eye(d) * 0.5
    return GaussianMixture(torch.arange(1, k + 1, dtype=torch.float64), mean, covariance)


@pytest.mark.parametrize("k", [1, 4, 16])
def test_assignment(k):
    scores = torch.randn(64, k, dtype=torch.float64)
    pi = torch.arange(1, k + 1, dtype=torch.float64)
    pi /= pi.sum()
    assignment = balanced_assignment(scores, pi)
    torch.testing.assert_close(
        assignment.sum(1), torch.ones(64, dtype=torch.float64), atol=1e-8, rtol=0
    )
    torch.testing.assert_close(assignment.sum(0), 64 * pi, atol=1e-8, rtol=0)


@pytest.mark.parametrize("objective", ["kl", "w2"])
@pytest.mark.parametrize("k", [1, 4])
def test_route_gradient(objective, k):
    ref = reference(k)
    state = MomentState.from_reference(ref)
    state.mean += 0.3
    state.second = ref.covariance + state.mean[:, :, None] * state.mean[:, None, :]
    route = Branch(ref, state, objective, ridge=0.02)
    x = torch.randn(64, ref.dimension)
    gradient, loss = route.feature_gradient(x, 0.99)
    assert gradient.shape == x.shape and np.isfinite(loss)
    assert torch.isfinite(gradient).all()
    old = state.mean.clone()
    route.commit()
    assert not torch.equal(old, state.mean)


def test_stop_gradient():
    ref = reference(1)
    state = MomentState.from_reference(ref)
    state.mean += 0.1
    state.second = ref.covariance + state.mean[:, :, None] * state.mean[:, None, :]
    x = torch.randn(17, ref.dimension, dtype=torch.float64, requires_grad=True)
    velocity = kl_velocity(x, torch.ones(17, 1), ref, state, 0.02)
    (gradient,) = torch.autograd.grad(regression_loss(x, velocity), x)
    torch.testing.assert_close(gradient, -2 * velocity / len(x))


@pytest.mark.parametrize("domain", ["imagenet", "t2i"])
@pytest.mark.parametrize("k", [1, 4])
def test_ema_score_timing(domain, k):
    ref = reference(k)
    state = MomentState.from_reference(ref)
    x = torch.randn(64, ref.dimension, dtype=torch.float64)
    from mgflow.distributions.assignment import assign

    assignment = assign(x, ref)
    mean, second = moments(x, assignment, ref.weight)
    expected = MomentState(state.mean.clone(), state.second.clone())
    if domain == "t2i":
        expected.update(mean, second, 0.99, domain)
    route = Branch(ref, state, ridge=0.02, domain=domain)
    velocity = route.velocity(x, 0.99)
    torch.testing.assert_close(
        velocity, kl_velocity(x, assignment, ref, expected, 0.02, domain), rtol=0, atol=0
    )
    torch.testing.assert_close(state.mean, expected.mean, rtol=0, atol=0)
    if domain == "imagenet":
        expected.update(mean, second, 0.99, domain)
    route.commit()
    torch.testing.assert_close(state.mean, expected.mean, rtol=0, atol=0)
    torch.testing.assert_close(state.second, expected.second, rtol=0, atol=0)


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "model.pth"
    save_checkpoint(path, model, 1000)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"model", "step"}
    copy = torch.nn.Linear(3, 2)
    assert load_checkpoint(path, copy) == 1000
    for a, b in zip(model.parameters(), copy.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_compiled_checkpoint(tmp_path):
    model = torch.nn.Linear(3, 2)
    compiled = torch.compile(model, backend="eager")
    path = tmp_path / "compiled.pth"
    save_checkpoint(path, compiled, 2)
    restored = torch.nn.Linear(3, 2)
    load_checkpoint(path, restored)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_config_recipes():
    from mgflow.config import Config

    for objective, components in (("kl", [1, 4, 16]), ("w2", [1, 4])):
        assert Config(objective=objective, components=components).validate().lr > 0
    with pytest.raises(ValueError):
        Config(objective="w2", components=[16]).validate()
    with pytest.raises(ValueError):
        Config(domain="t2i", objective="w2", components=[1, 4]).validate()


def test_warmup_labels():
    from mgflow.training.imagenet import warmup_labels

    labels = warmup_labels(torch.arange(50000), seed=1)
    torch.testing.assert_close(torch.bincount(labels), torch.full((1000,), 50))
    expected = torch.tensor(np.random.default_rng(1730).permutation(1000))
    torch.testing.assert_close(labels[:1000], expected, rtol=0, atol=0)


@pytest.mark.parametrize("objective,k", [("kl", 1), ("kl", 4), ("kl", 16), ("w2", 1), ("w2", 4)])
def test_streamed_warmup(objective, k):
    from mgflow.distributions.assignment import assign

    ref = reference(k, d=3)
    x = torch.randn(2057, ref.dimension, dtype=torch.float64)
    route = Branch(ref, MomentState.empty(ref), objective)
    for batch in x.split(257):
        route.warmup(batch)
    route.finalize_warmup()
    mass = torch.zeros_like(ref.weight)
    first, second = torch.zeros_like(ref.mean), torch.zeros_like(ref.covariance)
    for block in x.split(1024 if k > 1 else 257):
        r = assign(block, ref)
        mass += r.sum(0) if objective == "w2" else len(block) * ref.weight
        first += r.T @ block
        second += torch.einsum("bk,bi,bj->kij", r, block, block)
    torch.testing.assert_close(route.state.mean, first / mass[:, None], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        route.state.second, second / mass[:, None, None], atol=1e-13, rtol=1e-13
    )
    assert route.state.count == len(x)
