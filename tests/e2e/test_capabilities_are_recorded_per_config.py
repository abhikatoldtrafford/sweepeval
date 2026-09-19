"""The artifact must record which family was gated off for WHICH model.

The runtime learned to detect capabilities per model. The artifact did not:
`_config_payload` serialized no capability report and no per-row skip list,
and the run-level `skipped` was derived from the run-level report alone -- the
report belonging to whichever model discovery happened to pick.

So a sweep whose axis is the model would write "retrieval SKIPPED --
UNSUPPORTED" beside observations containing retrieval rows. An artifact that
contradicts its own observations is worse than one that omits the claim, and
a scorecard cannot evidence per-model gating from a file that does not record
it.
"""

from __future__ import annotations

from sweepeval.capabilities.detect import (
    Capability,
    CapabilityReport,
    CapabilityResult,
    Support,
)
from sweepeval.corpus.loader import load_corpus
from sweepeval.execute.planner import ConfigSpec, SweepPlan
from sweepeval.execute.sweep import ConfigResult, SweepResult, SweepStatus, _collect_skips
from sweepeval.report.stored import (
    StoredConfig,
    _config_payload,
    _Spec,
    aggregates_payload,
)


def report(**verdicts: Support) -> CapabilityReport:
    results = {}
    for name, support in verdicts.items():
        cap = Capability(name)
        results[cap] = CapabilityResult(
            capability=cap, support=support, method="fixture",
            confidence=1.0, evidence={},
        )
    return CapabilityReport(results=results)


def all_supported(**overrides: Support) -> CapabilityReport:
    verdicts = {c.value: Support.SUPPORTED for c in Capability}
    verdicts.update(overrides)
    return report(**verdicts)


def row(model: str, capabilities: CapabilityReport) -> ConfigResult:
    return ConfigResult(
        config=ConfigSpec(
            config_id=f"cfg-{model}", params={"model": model},
            system_prompt=None, system_prompt_variant="none",
        ),
        capabilities=capabilities,
    )


SEARCH = all_supported(retrieval=Support.SUPPORTED,
                       tool_calling=Support.UNSUPPORTED)
PLAIN = all_supported(retrieval=Support.UNSUPPORTED,
                      tool_calling=Support.SUPPORTED)


def test_a_rows_skip_list_follows_its_own_model() -> None:
    search = dict(row("gpt-5-search-api", SEARCH).skipped)
    plain = dict(row("gpt-4o-mini", PLAIN).skipped)

    assert "tool_integrity" in search and "retrieval" not in search
    assert "retrieval" in plain and "tool_integrity" not in plain


def test_the_payload_carries_the_rows_verdicts_and_skips() -> None:
    payload = _config_payload(row("gpt-5-search-api", SEARCH))

    assert payload["capabilities"]["retrieval"]["verdict"] == "SUPPORTED"
    assert payload["capabilities"]["tool_calling"]["verdict"] == "UNSUPPORTED"
    assert "tool_integrity" in dict(payload["skipped"])
    assert "retrieval" not in dict(payload["skipped"])


def _sweep(rows: list[ConfigResult]) -> SweepResult:
    result = SweepResult(
        run_id="r", profile="standard", runs=1, status=SweepStatus.COMPLETE,
        corpus=load_corpus("standard"), plan=SweepPlan(configs=()),
        capabilities=PLAIN,
    )
    result.configs.extend(rows)
    return result


def test_run_level_skipped_never_contradicts_a_row_that_scored_it() -> None:
    """The defect, stated as a property.

    `retrieval` ran for the search model, so the run-level list must not call
    it skipped -- even though the run-level capability report, built against
    the model discovery picked, says UNSUPPORTED.
    """
    result = _sweep([row("gpt-5-search-api", SEARCH), row("gpt-4o-mini", PLAIN)])
    _collect_skips(result)

    families = dict(result.skipped)
    assert "retrieval" not in families, (
        "claimed run-wide, but the search model scored it"
    )
    assert "tool_integrity" not in families, (
        "claimed run-wide, but the other model scored it"
    )


def test_a_family_no_config_could_score_is_still_reported_run_level() -> None:
    """I5 must not be weakened into silence by the intersection."""
    blind = all_supported(retrieval=Support.UNSUPPORTED,
                          tool_calling=Support.UNSUPPORTED)
    result = _sweep([row("a", blind), row("b", blind)])
    _collect_skips(result)

    families = dict(result.skipped)
    assert "retrieval" in families and "tool_integrity" in families
    assert "unsupported" in families["retrieval"].lower()


def test_a_sweep_with_no_configs_still_reports_from_the_run_level_report() -> None:
    result = _sweep([])
    _collect_skips(result)
    assert "retrieval" in dict(result.skipped)


def test_the_verdicts_survive_a_write_and_read(tmp_path) -> None:
    """Round trip through disk, not through `json.dumps`.

    Re-serializing the payload I had just built proved nothing: it never
    touched a restored row, which is the only one whose `capabilities` is a
    plain dict rather than a report. That branch is where a re-report of a
    stored run would have crashed.
    """
    result = _sweep([row("gpt-5-search-api", SEARCH), row("gpt-4o-mini", PLAIN)])
    payload = aggregates_payload(result)

    by_model = {c["params"]["model"]: c for c in payload["configs"]}
    assert by_model["gpt-5-search-api"]["capabilities"]["retrieval"]["verdict"] == (
        "SUPPORTED"
    )
    assert by_model["gpt-4o-mini"]["capabilities"]["retrieval"]["verdict"] == (
        "UNSUPPORTED"
    )
    assert "tool_integrity" in dict(by_model["gpt-5-search-api"]["skipped"])
    assert "retrieval" in dict(by_model["gpt-4o-mini"]["skipped"])

    restored = [
        StoredConfig(
            config=_Spec(
                config_id=c["config_id"], params=c["params"],
                system_prompt_variant="none", _label=c["config_id"],
            ),
            capabilities=c["capabilities"],
            skipped=tuple(tuple(s) for s in c["skipped"]),
        )
        for c in payload["configs"]
    ]
    again = {
        r.config.params["model"]: _config_payload(r) for r in restored
    }
    assert again["gpt-5-search-api"]["capabilities"]["retrieval"]["verdict"] == (
        "SUPPORTED"
    )
    assert "tool_integrity" in dict(again["gpt-5-search-api"]["skipped"])
