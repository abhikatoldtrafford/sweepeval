# sweepeval - quick sweep

20260912T182334743-f6f941 - 6 configuration(s) - 3 runs - **COMPLETE**

> Intervals at `quick` are valid but wide, so few pairs will separate. **Not gate-eligible** - use `--profile standard` for decisions or gating.

| config | config_repeatability | context_retention_auc | error_rate | fact_recall | guardrail_pass_rate | invariance | latency_ms | security_pass_rate | semantic_stability | target_determinism_at_temp0 | tokens_out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| temperature=0.0 sys=none | 1 [0.679, 1] | 1 [0.679, 1] | 0 [0, 0.104] | 1 [0.679, 1] | 1 [0.596, 1] | n/a | 0.6781 [0.655, 0.701] | 1 [0.679, 1] | 1 [0.679, 1] | 1 [0.679, 1] | 12 [7.7, 16.6] |
| temperature=1.0 sys=none | 0.2 [0, 0.521] | 1 [0.679, 1] | 0 [0, 0.104] | 1 [0.679, 1] | 1 [0.596, 1] | n/a | 0.6871 [0.653, 0.729] | 1 [0.679, 1] | 0.4667 [0.213, 0.739] | 1 [0.679, 1] | 12.65 [8.37, 17.3] |
| temperature=0.0 sys=terse_neutral | 1 [0.679, 1] | 1 [0.679, 1] | 0 [0, 0.104] | 1 [0.679, 1] | 1 [0.596, 1] | n/a | 0.628 [0.611, 0.645] | 1 [0.679, 1] | 1 [0.679, 1] | 1 [0.679, 1] | 12 [7.7, 16.6] |
| temperature=1.0 sys=terse_neutral | 0.2 [0, 0.521] | 1 [0.679, 1] | 0 [0, 0.104] | 1 [0.679, 1] | 1 [0.596, 1] | n/a | 0.7032 [0.628, 0.829] | 1 [0.679, 1] | 0.4667 [0.213, 0.739] | 1 [0.679, 1] | 12.65 [8.37, 17.3] |
| temperature=0.0 sys=verbose_strict_with_guardrails | 1 [0.679, 1] | 1 [0.679, 1] | 0 [0, 0.104] | 1 [0.679, 1] | 1 [0.596, 1] | n/a | 0.9928 [0.64, 1.66] | 1 [0.679, 1] | 1 [0.679, 1] | 1 [0.679, 1] | 12 [7.7, 16.6] |
| temperature=1.0 sys=verbose_strict_with_guardrails | 0.2 [0, 0.521] | 1 [0.679, 1] | 0 [0, 0.104] | 1 [0.679, 1] | 1 [0.596, 1] | n/a | 0.6753 [0.646, 0.709] | 1 [0.679, 1] | 0.4667 [0.213, 0.739] | 1 [0.679, 1] | 12.65 [8.37, 17.3] |

## frontier

6 non-dominated configuration(s) in 1 tied cluster(s), alpha 0.05 family-wise.

- **cluster 1**: cfg-00, cfg-01, cfg-02, cfg-03, cfg-04, cfg-05

## SKIPPED

- **degradation** - not_implemented_in_v0.1: concurrency ramp, long inputs and induced tool failures land in v0.2 (spec section 11, family 8)
- **retrieval** - retrieval=UNSUPPORTED (citation probe)
- **tool_integrity** - tool_calling=UNSUPPORTED (tool probe)
