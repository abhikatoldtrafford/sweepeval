# sweepeval — single-configuration evaluation

- **target** `https://mock.test/v1/chat/completions`
- **shape** `openai.chat_completions`
- **type** BARE_MODEL
- **profile** quick (40 units, 60 calls/run)
- **run** `20260913T032839015-6475fc`

## Metrics

| metric | value | 95% interval | clusters | flags |
|---|---:|---:|---:|---|
| `config_repeatability` | 1 | [0.679, 1] | 10 | INDICATIVE |
| `context_retention_auc` | 1 | [0.679, 1] | 10 | INDICATIVE |
| `error_rate` | 0 | [0, 0.104] | 40 | INDICATIVE |
| `fact_recall` | 1 | [0.679, 1] | 10 | INDICATIVE |
| `guardrail_pass_rate` | 1 | [0.596, 1] | 7 | LOW_N, INDICATIVE |
| `invariance` | — | no valid interval | 0 | INDICATIVE, LOW_N, NO_VALID_INTERVAL |
| `latency_ms` | 0.6615 | [0.64, 0.686] | 40 | INDICATIVE |
| `security_pass_rate` | 1 | [0.679, 1] | 10 | INDICATIVE |
| `semantic_stability` | 1 | [0.679, 1] | 10 | INDICATIVE |
| `target_determinism_at_temp0` | 1 | [0.679, 1] | 10 | INDICATIVE |
| `tokens_out` | 11.95 | [7.65, 16.6] | 40 | INDICATIVE |

## No variable axes discovered

This is a single configuration, not a sweep. Candidate axes and why each was rejected:

- `model` — 0 model identifier(s) discovered
- `temperature` — sampling-effect test not run in single-config evaluation

## Skipped

- `degradation` — not_implemented_in_v0.1: concurrency ramp, long inputs and induced tool failures land in v0.2 (spec section 11, family 8)
- `retrieval` — retrieval=UNSUPPORTED (citation probe)
- `tool_integrity` — tool_calling=UNSUPPORTED (tool probe)

## Assumptions you can correct

- `target.target_type` (low) — BARE_MODEL — it gates which results may be compared (I6)
