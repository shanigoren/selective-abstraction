# Selective Abstraction

Code accompanying **When Should LLMs Be Less Specific? Selective Abstraction for Reliable Long-Form Text Generation**.

Accepted to **NeurIPS 2026**.

Atom-wise Selective Abstraction (Aw-SA) replaces uncertain claims with less specific alternatives. A confidence threshold controls the retained detail. The generation procedure uses the generating model throughout; Wikipedia correctness and Wikidata coverage evaluation are separate offline measurements.

This release contains the generation components, the paper's baseline implementations, offline evaluators, and risk–coverage computations. It includes no generated responses, annotations, raw results, cached evaluations, model weights, or experiment notebooks. Threshold calibration and transfer are held for a separate review and are not included.

## Setup

Use Python 3.12 for the generation and analysis examples. Run commands from the repository root. Serve an open-weight model through an OpenAI-compatible endpoint in a separate environment, or use the OpenAI API for GPT-5.2.

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements-core.txt -r requirements-analysis.txt
```

For a local endpoint, set `VLLM_BASE_URL` and, if required by your server, `VLLM_API_KEY`. The default endpoint is `http://localhost:8000/v1`. For GPT-5.2, set `OPENAI_API_KEY` in your shell. Keep credentials outside the repository.

The provided model aliases are `gpt-5.2`, `gpt-oss-120b`, `gpt-oss-20b`, `qwen3-30b-thinking`, and `qwen3-4b-thinking`. Configure your server's served-model name to match the selected alias. Generation parameters are in `pipeline/utils/model_generation_params.json`.

## Run Aw-SA

```bash
python pipeline/run_sa.py --model gpt-oss-120b --input "Describe a topic of your choice." --generation-prompt longfact --theta 70 --output-dir outputs/example
```

For a biography, use `--generation-prompt factscore` and pass the person's name as `--input`. Use `--input-file` for a UTF-8 prompt file. `--selection redaction` applies deterministic removal instead of abstraction selection.

The command retains intermediate files in the selected output directory and reconstructs the selected claims using `recon_rf_v6`, the prompt used in the reconstruction validation. All model calls use the same model and generation-parameter mapping. No correctness or coverage evaluator is invoked. These commands perform model inference; the offline tests below do not.

Selection follows the experimental implementation: the first abstraction with confidence **at least** the threshold is retained. See [reproducibility notes](docs/reproducibility.md) for the boundary convention and the distinction between inference redaction and the historical evaluation baseline.

## Prepare benchmark inputs

The input manifest fixes the 183 FactScore entities and 76 LongFact-Objects prompts, with two prompts from each of 38 subjects. It stores content hashes and selection IDs, not model outputs.

Obtain the labeled `prompt_entities.txt` from the [official FactScore data release](https://github.com/shmsw25/FActScore#download-the-data). Then run:

```bash
python scripts/prepare_datasets.py --factscore-entities INPUT_ENTITIES.txt
```

The script validates the FactScore list, downloads LongFact inputs from a pinned upstream revision, checks every selected prompt hash, and writes the inputs under the ignored `datasets/` directory. It refuses mismatching inputs. It does not resample the cohort.

## Baselines and benchmark generation

`pipeline/main.py` retains the benchmark generation workflow. Its output consists of generated text, atoms, abstractions where applicable, and confidence scores. Use `pipeline/run_sa.py` when a reconstructed final Aw-SA answer is required.

| Paper method | Entry point / experiment |
| --- | --- |
| Aw-SA | `pipeline/run_sa.py`; benchmark components: `--experiment sa` |
| Selective Redaction | `pipeline/run_sa.py --selection redaction` |
| Self-Revision | `pipeline/main.py --experiment double_pass` |
| Atom-wise Self-Revision | `pipeline/main.py --experiment atom_wise_self_revision` |
| Iterative Self-Revision | `pipeline/main.py --experiment iterative_self_revision --iterative_k 4` |
| Inline Abstraction | `pipeline/main.py --experiment single_pass` |
| DRAG | `pipeline/main.py --experiment sampling_baseline` |
| Chain-of-Verification | `pipeline/main.py --experiment cove` |

For example, after preparing inputs:

```bash
python pipeline/main.py --dataset factscore_labeled --generating_model gpt-oss-120b --experiment double_pass --count 1 --run_questions false --results_root outputs/benchmark
```

Read the DRAG seed/failure behavior and the benchmark workflow's reuse rules in [reproducibility notes](docs/reproducibility.md) before interpreting a rerun as a replication.

The appendix's threshold-prompt variants use `--experiment single_pass_theta` or `--experiment double_pass_theta` together with `--theta`. For the qualitative non-factual Aw-SA setting, use `pipeline/run_sa.py --generation-prompt non_factual` with your own input prompt.

## Offline evaluation and analysis

The Wikipedia correctness evaluator is under `correctness/wikipedia_factcheck_agent/`. The SPINACH-based Wikidata count evaluator is under `coverage/`. Each has its own dependency file. They require model endpoints and access to the relevant knowledge sources; they are not needed to generate an Aw-SA answer. See [evaluation instructions](docs/evaluation.md).

The `analysis` module consumes evaluated rows and computes the historical Aw-SA and redaction RC curves. It exposes full-range and shared-overlap AURC as separate operations:

```bash
python -m analysis curves INPUT_ROWS.csv outputs/curves.csv
python -m analysis aurc-full outputs/curves.csv outputs/aurc.json --curve sa
python -m analysis aurc-shared outputs/curves.csv outputs/shared-aurc.json --curve-a sa --curve-b hr
```

Input columns and the preserved filtering, deduplication, normalization, and failure conventions are documented in [reproducibility notes](docs/reproducibility.md).

## Offline checks

```bash
python -m unittest discover -s tests -v
python scripts/verify_release.py
```

Tests use invented inputs and mocked model responses. They check software behavior; they do not reproduce the paper's empirical results.

Run the release verifier on a clean copy before generating local outputs. It checks the exact file inventory and hashes and reports extra files, including ignored results. It is an additional publishing check, not a guarantee that future edits contain no secrets.

## License and attribution

Original release code is provided under [MIT](LICENSE). SPINACH-derived components retain Apache-2.0. Benchmark inputs retain their upstream terms; see [third-party notices](THIRD_PARTY.md) and `LICENSES/`.
