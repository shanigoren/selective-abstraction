# Reproducibility scope and conventions

## What is included

- Aw-SA generation stages and a new end-to-end command that joins those stages to deterministic selection and validated v6 reconstruction.
- The seven baseline implementations and their required prompts.
- Exact input-cohort IDs and hashes, with an input preparation script.
- The Wikipedia correctness evaluator and SPINACH/Wikidata count evaluator.
- Code-only RC and AURC computation extracted from the analysis implementation.

Threshold calibration and transfer are excluded pending a separate review. The saved calibration runner uses an interpolated quantile; it has not been validated as an implementation of the paper's stated order-statistic algorithm. This does not change the separate direct threshold sweep used for RC curves.

The release does not contain the manual annotation studies, their labels, individual judgments, or raw outputs. It does not reproduce the supplementary human/LLM-judge tables from a single command. Generation-latency experiments and supplementary annotation workflows are not bundled in this initial code package. The paper's threshold-prompt variants and non-factual Aw-SA template are included; no non-factual prompt collection or example responses are distributed.

## Generation

The end-to-end command uses the same generating model for generation, atomization, atom confidence, abstraction, abstraction confidence, and reconstruction. The original refusal-classification step is retained. It is an additional preprocessing call.

The reconstruction default is `recon_rf_v6`; its source prompt is retained verbatim. The earlier reconstruction prompt is preserved as a separately named non-default component. Prompt selection is not interchangeable when comparing results.

The new end-to-end command rejects incomplete or misaligned stage outputs before selecting claims. This is stricter than the historical benchmark validation, which tolerated some missing records. A rejected run is a failure to inspect, not an automatically repaired or resampled experiment.

Runtime selection uses `confidence >= theta`, as in the experimental code and the RC sweep. The paper's displayed selection equation uses a strict inequality. At a score exactly equal to the threshold, these conventions differ; the release preserves the implementation convention.

Model aliases are not checkpoint identifiers. Reproduction records should include the actual served checkpoint/revision, serving software version and flags, tokenizer version, reasoning effort, generation parameters, and provider model identifier. The repository does not claim to reconstruct missing historical checkpoint or serving metadata.

The original local-model completion helper uses seed 123. OpenAI-hosted calls do not use that seed. This difference is retained. Provider changes and nondeterminism can change regenerated responses.

The historical DRAG implementation requests five responses using that helper. It reuses the helper's seed and can omit a response if parsing/resolution fails. Thus a run can contain fewer than five successful samples, and fixed-seed local responses can coincide. Inspect the saved sampling stage before interpreting a comparison. No new sampling policy or repaired baseline result is claimed here.

Stage functions reuse existing outputs when their configured output path exists. Use a new output directory when changing prompts, model checkpoints, or generation parameters. Output paths in this release use portable `key=value` components rather than the original platform-specific naming convention.

## Input data

`configs/input_manifest.json` binds the paper cohort using IDs and SHA-256 content hashes. FactScore contains 183 entities. LongFact contains 76 prompts, two per subject across 38 subjects. LongFact is fetched from a pinned public revision and verified against the recorded selected prompt content.

The original sampler's current default is not used: it would select a different cohort size. The preparation script materializes the recorded selection instead of drawing a new sample.

## Evaluated-row schema

The RC functions consume rows with these fields:

| Field | Meaning |
| --- | --- |
| `entity` | Stable response/entity ID |
| `atom_id` | Original atom identifier within the response |
| `abstraction` | Candidate claim text |
| `confidence` | Atom confidence for index zero; abstraction confidence for later levels |
| `broad_count` | Reference entity-set count used in the analysis |
| `specific_count` | Conditioned entity-set count |
| `correctness` | `SUPPORTED`, `UNSUPPORTED`, or `FAILED` |

Rows must be ordered from most to least specific within each atom. Supply the same reference-count convention used by the intended analysis, including any broad-people override. The metrics module does not silently obtain a new count from a live knowledge source.

## Historical metric behavior

The reproduced row filters remove failed correctness labels, reference counts at most one, and negative condition counts. Information is `1 - log2(specific_count) / log2(broad_count)`, with the historical zero-count fallback to zero and clipping to `[0, 1]`.

Aw-SA filters candidates by confidence before taking the first remaining row for each atom. The historical redaction analysis groups after the validity filters and then thresholds the first surviving row. If an original atom's evaluation was removed, this historical analysis can retain a broader row. It is distinct from the strict original-atom redaction used by the inference command. The behavior is named explicitly and has not been silently corrected.

The analysis deduplicates exact claim strings within a response. Risk is one minus mean correctness over selected claims; it is claim-count weighted. Coverage is the sum of the selected information scores. The default sweep uses 101 integer thresholds from 0 through 100. Empty selections have undefined risk rather than an invented numerical score.

Two AURC conventions are kept separate:

- `aurc_full_range_historical` adds a zero-coverage anchor at the minimum-coverage risk and normalizes each curve by its own maximum coverage before trapezoidal integration.
- `aurc_shared_overlap` integrates only the common coverage range with linearly interpolated boundary risks and a shared maximum-coverage normalizer. It does not divide the area by the overlap width.

Do not compare these two outputs as if they were the same statistic. The module also exposes the distinct single-plot and grid-plot coverage normalizers used by the historical analysis.

Normalization and AURC require valid positive coverage. Empty or all-zero inputs raise an error rather than producing an infinite or undefined area. These input checks do not change the formulas on valid inputs.

## Verification limits

Offline tests check selection, output wiring, and numeric behavior using invented data. Input preparation checks the exact cohort content. These are software and input-integrity checks, not new evidence about model quality.

Full model inference, the live Wikipedia/Wikidata evaluators, and empirical reproduction require the corresponding model endpoints and compute. Wikipedia/Wikidata content and hosted model behavior can change. No raw result set is included to hide or bypass these dependencies.
