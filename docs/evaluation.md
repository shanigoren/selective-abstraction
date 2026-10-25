# Offline evaluation

Run generation first. Correctness and coverage consume the generated candidate claims and counting questions. They are separate measurements and are never required by `pipeline/run_sa.py`.

Use separate environments for the two evaluators. Their direct dependency versions come from the recorded research environments; these files are not a complete historical lockfile. A model server is a separate dependency.

## Correctness

Install `requirements-correctness.txt`. Configure `SA_MODEL_BASE_URL` for an OpenAI-compatible gpt-oss-120b evaluator, and set `SA_MODEL_API_KEY` only if your endpoint requires authentication. Set `SA_WIKIMEDIA_USER_AGENT` to an appropriate identifying user-agent value for your deployment.

```bash
python correctness/wikipedia_factcheck_agent/agent.py --input_path INPUT_ABSTRACTIONS.json --model gpt-oss-120b --reasoning_effort high --max_tokens 64000 --no_cache
```

The input is the candidate abstraction JSON produced by the pipeline. The evaluator writes correctness labels and logs beside that input. It searches Wikipedia and labels claims supported or unsupported; failures remain visible in the generated artifacts.

## Coverage

Install `requirements-coverage.txt` in a separate environment. A Redis server and access to the Wikidata Query Service are also required. The original research coverage environment used Python 3.10. The release preserves the SPINACH controller and counting prompts.

Set `SA_MODEL_BASE_URL` to the coverage evaluator's endpoint and `VLLM_API_KEY` if required. Redis defaults to localhost. `SA_REDIS_SERVER` may specify the Redis server executable if it is not on `PATH`.

```bash
python coverage/coverage.py --questions_path INPUT_QUESTIONS.json --backend legacy --model gpt-oss-120b --reasoning_effort high --max_response_tokens 30000 --temperature 1.0 --top_p 1.0 --top_k 0 --no_cache
```

`INPUT_QUESTIONS.json` is produced by the pipeline's `run_questions` stage. The evaluator obtains broad and conditioned entity counts. Conversion to information and risk–coverage statistics belongs to the separate `analysis` module.

## Join the evaluator outputs

Use the explicit-path assembler to join one entity's counts, correctness labels, and confidence scores into the RC row schema:

```bash
python -m analysis.assemble_rows --counts INPUT_COUNTS.json --correctness INPUT_CORRECTNESS.json --confidence INPUT_CONFIDENCE.json --entity ENTITY_ID --output outputs/rows.json --broad-people-count-override none
python -m analysis curves outputs/rows.json outputs/curves.csv
```

The assembler requires an explicit choice for the broad-people count override: pass a count from your own recorded evaluation, or `none` to retain per-row counts. The historical analysis used a shared override, but its measured value is not distributed with the code. Record the choice and do not mix these conventions within a claimed replication. The assembler records the atom-level confidence at abstraction index zero. It rejects mismatched or malformed records instead of silently truncating or skipping them, so failed joins must be resolved explicitly before analysis.

For multiple entities, assemble one row file per entity and pass additional row files with repeated `--additional-rows` options. Keep entity IDs unique. The curves are then computed over the combined rows, rather than by averaging per-entity risks.

## Cache and output handling

The global result cache is disabled by default. Historical cache keys do not include the full evaluator configuration. Reusing them across models, prompts, or decoding settings can reuse incompatible results; keep caching disabled for fresh comparisons.

If you explicitly enable caching, cache files are created under `outputs/cache/`; set `SA_CACHE_DIR` to use a different local directory. `--verbose` and the correctness evaluator's `--skip_failed` are valueless true flags; omit them for false.

Outputs and caches are local artifacts and are ignored by Git. Use fresh output locations for changed configurations. The public repository contains no stored Wikipedia evidence, Wikidata responses, label files, or raw model outputs.
