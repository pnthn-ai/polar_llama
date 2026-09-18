"""CI smoke test for the local backend on a machine WITHOUT mlx.

This is the canary the Linux CI job runs (``-m "local and not local_gpu"``):
it proves the local-backend package imports and its ``in_process`` logic runs
end-to-end using the dependency-free ``FakeEngine`` -- no ``mlx``/``mlx-lm``,
no GPU, no network. If any of these fail, the ``[local]`` extra has leaked a
mandatory mlx import into the base package.

The real ``MlxBatchEngine`` path is deliberately NOT exercised here; it lives
behind the optional ``[local]`` extra and requires Apple-Silicon GPU hardware
(marked ``local_gpu`` elsewhere and excluded from CI).
"""

from __future__ import annotations

import sys

import polars as pl
import pytest

from helpers import assert_import_stays_mlx_free

pytestmark = pytest.mark.local


def test_import_local_does_not_import_mlx():
    """Importing ``polar_llama.local`` must never eagerly import mlx/mlx-lm.

    Checked in a fresh interpreter. Asserting ``"mlx" not in sys.modules``
    in-process only holds on a machine where mlx is absent AND no earlier test
    imported it, so it quietly stopped testing anything on an Apple-silicon box
    with the ``[local]`` extra -- the machine most able to break the invariant.
    """
    import polar_llama.local as local

    assert local is not None
    assert_import_stays_mlx_free(["polar_llama.local"])


def test_mlx_availability_is_reported_consistently():
    """``HAS_MLX`` and ``is_mlx_available()`` must agree with the environment.

    The previous form asserted both were ``False``, which is a fact about a
    machine without the ``[local]`` extra rather than about this code -- it
    failed wherever mlx is genuinely installed. What is worth pinning is that
    the lazily-computed ``HAS_MLX`` tracks ``is_mlx_available()`` and that both
    return real booleans.
    """
    from polar_llama.local import HAS_MLX, is_mlx_available

    try:
        import mlx_lm  # noqa: F401

        expected = True
    except ImportError:
        expected = False

    assert is_mlx_available() is expected
    assert HAS_MLX is expected


def test_require_mlx_raises_helpful_error(monkeypatch):
    """``require_mlx`` should point users at the optional extra, not crash raw.

    The error path is forced by poisoning ``sys.modules`` (a ``None`` entry
    makes ``import`` raise ``ImportError``), so this exercises the message on
    every machine instead of only where mlx happens to be missing.
    """
    from polar_llama.local import require_mlx

    monkeypatch.setitem(sys.modules, "mlx_lm", None)

    with pytest.raises(ImportError, match=r"polar-llama\[local\]"):
        require_mlx()


def test_require_mlx_returns_the_module_when_available():
    """The success path, on a machine that actually has the extra."""
    from polar_llama.local import is_mlx_available, require_mlx

    if not is_mlx_available():
        pytest.skip("mlx-lm not installed")

    assert require_mlx().__name__ == "mlx_lm"


def test_inference_local_in_process_fake_row_order(monkeypatch):
    """FakeEngine path of ``inference_local(engine="in_process")`` runs on CPU.

    Forces the dependency-free FakeEngine via the ``POLAR_LLAMA_LOCAL_ENGINE``
    override so the whole ``in_process`` map_batches UDF is exercised without
    importing mlx. Asserts a String column returned in ORIGINAL ROW ORDER.
    """
    monkeypatch.setenv("POLAR_LLAMA_LOCAL_ENGINE", "fake")

    from polar_llama.local.engine import clear_registry
    from polar_llama.local.expr import inference_local

    clear_registry()
    try:
        prompts = [f"prompt-{i}" for i in range(25)]
        df = pl.DataFrame({"prompt": prompts})
        result = df.with_columns(
            answer=inference_local(
                pl.col("prompt"),
                model="ci-smoke-model",
                engine="in_process",
                max_tokens=512,
            )
        )

        # Deterministic FakeEngine output, one row per input, original order.
        assert result.schema["answer"] == pl.String
        assert result["answer"].to_list() == [f"echo:{p}" for p in prompts]
    finally:
        clear_registry()
