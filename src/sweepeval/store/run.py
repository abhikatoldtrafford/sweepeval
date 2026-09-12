"""Run directory layout and the ``Store`` facade (spec §6.2).

``Store`` is the only way to obtain a writer. Every writer it hands out is
already wired to the run's redactor, which is what makes the no-secrets
contract test exhaustive rather than a spot check: there is no other
constructor path for a caller to reach past.
"""

from __future__ import annotations

from pathlib import Path

from sweepeval.schema.call import Call
from sweepeval.schema.observation import Observation
from sweepeval.store.blob import BlobStore
from sweepeval.store.jsonl import AppendOnlyLog
from sweepeval.store.redaction import Redactor
from sweepeval.store.state import RunState, new_run_id

__all__ = ["Store", "new_run_id"]


class Store:
    """The artifact store for one run."""

    def __init__(
        self,
        root: Path,
        run_id: str,
        redactor: Redactor,
        *,
        store_text: bool = True,
    ) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.redactor = redactor
        self.run_dir = self.root / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "state").mkdir(exist_ok=True)

        self.calls: AppendOnlyLog[Call] = AppendOnlyLog(
            self.run_dir / "calls.jsonl", Call, redactor
        )
        self.observations: AppendOnlyLog[Observation] = AppendOnlyLog(
            self.run_dir / "observations.jsonl", Observation, redactor
        )
        self.blobs = BlobStore(
            self.run_dir / "blobs", redactor, store_text=store_text
        )

    # --- derived and metadata paths ---------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def plan_path(self) -> Path:
        return self.run_dir / "plan.json"

    @property
    def aggregates_path(self) -> Path:
        return self.run_dir / "aggregates.json"

    @property
    def frontier_path(self) -> Path:
        return self.run_dir / "frontier.json"

    def report_path(self, fmt: str) -> Path:
        return self.run_dir / f"report.{fmt}"

    # --- resume state ------------------------------------------------------

    def state_for(self, config_id: str) -> RunState:
        return RunState(self.run_dir / "state" / f"{config_id}.json")

    @property
    def state(self) -> RunState:
        """Run-level state, for callers not yet scoped to a config."""
        return RunState(self.run_dir / "state" / "_run.json")

    def provenance(self) -> dict[str, str]:
        """``derived_from`` map for anything computed off this run's logs."""
        return {
            "calls.jsonl": self.calls.content_hash(),
            "observations.jsonl": self.observations.content_hash(),
        }
