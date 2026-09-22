# Result files and interpretation

These files are checked-in outputs from exploratory engineering runs. They are useful for reproducing calculations and inspecting failure modes, but they are not publication-grade evidence: sample sizes are small, most questions were run once, and several evaluations use one LLM judge.

Run the deterministic summary with:

```bash
python benchmark/summarize_results.py
```

## Canonical files

| Purpose | File | Notes |
|---|---|---|
| Base prompt comparison | `full_system.json`, `single_llm.json`, `original_devil.json` | 10 matching questions per system; historical multi-agent runs use cosine routing |
| NLI routing behavior | `nli_detection_fixed3rounds_baseline.json` | 7 matching questions; 4 reached multiple rounds |
| Later mixed NLI runs | `nli_detection.json` | Contains resumed runs from different question slices; do not treat its `meta.total_questions` as the dataset size |
| Adaptive prompt comparison | `type_comparison.json` | Contains validation sentinels; use matched valid-question filtering |
| Historical key factors | `historical/results.json` | Qwen-judged 0/1/2 key-factor mention scores |
| Hedge/commitment labels | `prohibition_analysis.json` | Single-judge categorical labels |
| General quality rubric | `quality_scores.json` | Contains legacy entries and later entries with explicit judge metadata |

## Validation failures

`type_comparison.json` contains 23 sentinel positions across 11 system runs, affecting seven unique questions: q4, q5, q40, q79, q80, q102, and q103. The original judge scores for those outputs remain in the raw file for auditability, but headline comparisons must exclude the affected question from both systems.

After matched-pair exclusion, the available sample sizes are binary n=8, values-based n=9, and context-dependent n=16 per system. These post-validation summaries are descriptive only.

## Historical evaluation wording

The historical rubric is:

- 0: missed the key factor
- 1: partially or tangentially mentioned it
- 2: clearly identified it as critical

No checked-in response received a score of 2. Therefore the reported 40%/60% values are **partial-or-better key-factor mention rates**, not decision accuracy. The scores were produced by an LLM judge and are not judge-independent ground truth.

## Provenance gaps

Older result files do not consistently record model version, backend, prompt hash, random seed, judge, or git commit. New runs should preserve at least:

- UTC run timestamp and git commit
- model provider and exact model identifier
- divergence mode and maximum rounds
- question file and selected IDs
- judge provider/model and rubric version
- validation failure counts

Raw files are retained rather than silently rewritten. Documentation uses cautious language where provenance is incomplete.

The current production graph defaults to NLI. Benchmark builders pin their
detector explicitly: named cosine ablations remain cosine, while
`nli_detection` remains NLI. This prevents an environment-default change from
silently changing the meaning of a future experiment.
