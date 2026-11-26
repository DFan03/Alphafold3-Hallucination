"""tmol FastRelax client for af3_hall (`--fastrelax`): relax every AF3-predicted state BEFORE it is scored.

Why: AF3 coordinates are not at a force-field minimum (side-chain clashes, strained contacts), so any energy-, contact-,
H-bond- or SASA-based binder metric computed on the raw model is noisy. Method = tmol 0.2.0 FastRelax (beta2016, GPU), in a
persistent worker process in a separate torch env (the AF3 env is JAX-only), see relax_tmol_worker.py for the protocol and the reasons
Rosetta was not chosen. The relaxed CIF replaces the AF3 CIF for everything downstream (scoring, proposals, saved CIFs);
the raw AF3 CIF is kept next to it.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKER = Path(__file__).with_name("relax_tmol_worker.py")


class RelaxError(RuntimeError):
    """Relax failed and relax_on_fail == 'raise'."""


class TmolRelaxer:
    """Persistent tmol worker. `relax()` returns a meta dict (ok, E_before, E_after, ca_rmsd, relax_sec, ...)."""

    def __init__(
        self,
        *,
        python: str | None = None,
        device: str = "cuda:0",
        repeats: int = 1,
        scope: str = "all",
        shell: float = 8.0,
        cartesian: bool = True,
        on_fail: str = "raise",
        timeout_s: float = 900.0,
        cmd_prefix: Sequence[str] = (),
    ) -> None:
        if scope not in ("all", "shell"):
            raise ValueError(f"relax_scope must be 'all' or 'shell', got {scope!r}")
        if on_fail not in ("raise", "skip"):
            raise ValueError(f"relax_on_fail must be 'raise' or 'skip', got {on_fail!r}")
        # interpreter of an env that has torch + tmol 0.2.0 + the RFD4-Proteina `rfproteina` package (see docs/fastrelax.md)
        self.python = python or os.environ.get("AF3_HALL_RELAX_PYTHON", "python")
        self.device = device
        self.repeats, self.scope, self.shell, self.cartesian = int(repeats), scope, float(shell), bool(cartesian)
        self.on_fail, self.timeout_s = on_fail, float(timeout_s)
        self.cmd_prefix = list(cmd_prefix)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.n_ok = self.n_fail = 0
        self.total_relax_sec = 0.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "TmolRelaxer":
        return cls(
            python=cfg.get("relax_python") or None,
            device=str(cfg.get("relax_device", "cuda:0")),
            repeats=int(cfg.get("relax_repeats", 1)),
            scope=str(cfg.get("relax_scope", "all")),
            shell=float(cfg.get("relax_shell", 8.0)),
            cartesian=bool(cfg.get("relax_cartesian", True)),
            on_fail=str(cfg.get("relax_on_fail", "raise")),
            timeout_s=float(cfg.get("relax_timeout_s", 900)),
        )

    # ---- process management ----
    def _start(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            cmd = [*self.cmd_prefix, self.python, str(WORKER), "--device", self.device]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1
            )
        return self._proc

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
                self._proc.wait(timeout=20)
            except Exception:
                self._proc.kill()
        self._proc = None

    def _call(self, req: dict[str, Any]) -> dict[str, Any]:
        proc = self._start()
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        box: list[str] = []
        th = threading.Thread(target=lambda: box.append(proc.stdout.readline()), daemon=True)  # type: ignore[union-attr]
        th.start()
        th.join(self.timeout_s)
        if th.is_alive() or not box or not box[0].strip():
            proc.kill()
            self._proc = None
            raise TimeoutError(f"tmol worker gave no reply within {self.timeout_s:.0f}s (or died)")
        return json.loads(box[0])

    # ---- public API ----
    def relax(self, in_cif: str | Path, out_cif: str | Path, *, focus_chains: Sequence[str] = ()) -> dict[str, Any]:
        """Relax `in_cif` -> `out_cif`. On failure: raise RelaxError, or (relax_on_fail='skip') return ok=False meta."""
        req = {
            "in_cif": str(in_cif), "out_cif": str(out_cif), "repeats": self.repeats, "cartesian": self.cartesian,
            "scope": self.scope, "shell": self.shell, "focus_chains": list(focus_chains),
        }
        t0 = time.time()
        with self._lock:
            try:
                meta = self._call(req)
            except Exception as exc:  # worker died / timeout / bad JSON: one restart-and-retry
                self.close()
                try:
                    meta = self._call(req)
                except Exception as exc2:
                    meta = {"ok": False, "error": f"{type(exc2).__name__}: {exc2} (first: {exc})"}
        meta["wall_sec"] = round(time.time() - t0, 3)
        if meta.get("ok") and Path(out_cif).exists():
            self.n_ok += 1
            self.total_relax_sec += float(meta.get("relax_sec", 0.0))
            return meta
        self.n_fail += 1
        if self.on_fail == "raise":
            raise RelaxError(f"tmol relax failed for {in_cif}: {meta.get('error')}")
        meta["ok"] = False
        meta["skipped"] = True
        return meta

    def summary(self) -> str:
        return (
            f"fastrelax(tmol, scope={self.scope}, repeats={self.repeats}): ok={self.n_ok} fail={self.n_fail} "
            f"relax_total={self.total_relax_sec:.1f}s"
        )
