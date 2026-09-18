"""Shared test helpers.

Not a test module (pytest only collects ``test_*.py``), just the two pieces
several suites need: a mock HTTP server whose listen backlog the Rust fan-out
cannot overflow, and a way to assert import hygiene that works on a machine
where the optional dependency is actually installed.
"""

from __future__ import annotations

import subprocess
import sys
from http.server import ThreadingHTTPServer
from typing import Sequence

#: Optional Apple-silicon dependencies that must never be imported eagerly.
MLX_MODULES = ("mlx", "mlx_lm", "mlx_embeddings")


class BacklogHTTPServer(ThreadingHTTPServer):
    """A ``ThreadingHTTPServer`` sized for the Rust client's fan-out.

    ``socketserver.TCPServer.request_queue_size`` defaults to **5**, but the
    Rust client opens up to ``POLAR_LLAMA_MAX_CONCURRENCY`` (64 by default)
    connections at once. Connections past the backlog are refused by the kernel
    before the accept loop ever sees them, which surfaces as::

        Request Error: error sending request for url (http://127.0.0.1:PORT/...)

    on a *subset* of rows -- 6 of 8 succeeding, say. Whether it happens at all
    depends on machine timing, so it comes and goes between runs and reads
    exactly like a library regression rather than a test-harness limit.

    128 leaves headroom over the default concurrency. Raise it if a suite ever
    drives more connections than that at one server.
    """

    request_queue_size = 128


def run_in_fresh_interpreter(code: str) -> str:
    """Run ``code`` in a new interpreter and return its stdout.

    A subprocess is the point: ``sys.modules`` in the *current* process is
    shared with every test that ran before, so an in-process import-hygiene
    assertion silently depends on test ordering.
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed ({result.returncode}):\n{result.stderr}"
    )
    return result.stdout.strip()


def _leak_probe(body: str) -> str:
    return (
        f"{body}\n"
        "import sys\n"
        f"print(','.join(m for m in {MLX_MODULES!r} if m in sys.modules))\n"
    )


def assert_stays_mlx_free(body: str, *, what: str) -> None:
    """Assert that running ``body`` in a fresh interpreter never imports mlx.

    Checked out-of-process so the assertion means the same thing on a Linux CI
    runner without the ``[local]`` extra and on an Apple-silicon dev machine
    that has it installed. The in-process form (``assert "mlx" not in
    sys.modules``) can only pass on a machine where mlx is absent *and* no
    earlier test imported it -- so it silently stopped testing anything on the
    one machine most able to break the invariant.
    """
    leaked = run_in_fresh_interpreter(_leak_probe(body))
    assert not leaked, f"{what} eagerly imported: {leaked}"


def assert_import_stays_mlx_free(modules: Sequence[str]) -> None:
    """Assert that importing ``modules`` does not pull mlx in with them."""
    imports = "\n".join(f"import {m}" for m in modules)
    assert_stays_mlx_free(imports, what=f"importing {', '.join(modules)}")
