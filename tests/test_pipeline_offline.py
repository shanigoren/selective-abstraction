import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


RELEASE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = RELEASE_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

# Permit source imports even when the optional SDK is not installed. No client
# is instantiated in these tests.
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _NoNetworkOpenAI:
        def __init__(self, *args, **kwargs):
            raise AssertionError("The offline test must not construct an SDK client.")

    openai_stub.OpenAI = _NoNetworkOpenAI
    sys.modules["openai"] = openai_stub

import reconstruction
import run_sa
from selection import select_statements


class _FakeCompletion:
    def __init__(self, text):
        self._text = text

    def to_dict(self):
        return {
            "choices": [{"message": {"content": self._text}}],
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 1,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }


class _QueuedCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected model completion call.")
        return _FakeCompletion(self.responses.pop(0))


class _FakeClient:
    def __init__(self, responses):
        self.chat = types.SimpleNamespace(
            completions=_QueuedCompletions(responses)
        )


class SelectionTests(unittest.TestCase):
    def test_sa_uses_atom_confidence_for_index_zero_and_exact_boundary(self):
        confidence_data = [
            {
                "atom": "[A] [specific].",
                "confidence": 80,
                "abstractions": [
                    {"index": 0, "abstraction": "[A] [specific].", "confidence": 1},
                    {"index": 1, "abstraction": "[A] [general].", "confidence": 99},
                ],
            },
            {
                "atom": "[B] [specific].",
                "confidence": 20,
                "abstractions": [
                    {"index": 0, "abstraction": "[B] [specific].", "confidence": 99},
                    {"index": 1, "abstraction": "[B] [general].", "confidence": 80},
                ],
            },
            {
                "atom": "[A duplicate] [specific].",
                "confidence": 80,
                "abstractions": [
                    {"index": 0, "abstraction": "[A] [specific].", "confidence": 0},
                ],
            },
        ]
        self.assertEqual(
            select_statements(confidence_data, theta=80, mode="sa"),
            ["[A] [specific].", "[B] [general]."],
        )

    def test_redaction_uses_original_atom_confidence_and_deduplicates(self):
        confidence_data = [
            {"atom": "[A] [fact].", "confidence": 50, "abstractions": []},
            {"atom": "[B] [fact].", "confidence": 49, "abstractions": []},
            {"atom": "[A] [fact].", "confidence": 90, "abstractions": []},
        ]
        self.assertEqual(
            select_statements(confidence_data, theta=50, mode="redaction"),
            ["[A] [fact]."],
        )


class ReconstructionTests(unittest.TestCase):
    def test_all_abstention_writes_empty_response_without_completion(self):
        prompts_dir = PIPELINE_DIR / "prompts"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            with mock.patch.object(
                reconstruction,
                "get_model_completion",
                side_effect=AssertionError("No completion is allowed for all-abstention."),
            ):
                output_dir = reconstruction.run_reconstruction(
                    client=object(),
                    model="gpt-oss-20b",
                    original_prompt="Synthetic question",
                    statements=[],
                    input_dir=temp_dir,
                    prompts_dir=str(prompts_dir),
                    generation_params={"temperature": 1.0},
                    theta=90,
                    sys_prompt="recon_rf_v6",
                )
            response_path = Path(output_dir) / "response.json"
            response = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertEqual(response["choices"][0]["message"]["content"], "")
            self.assertTrue(response["config"]["abstained"])
            self.assertEqual(response["config"]["sys_prompt"], "recon_rf_v6")
            self.assertEqual(
                run_sa._validate_reconstruction_output(
                    output_dir, statements=[], model="gpt-oss-20b"
                ),
                Path(output_dir),
            )


class StrictStageValidationTests(unittest.TestCase):
    def test_missing_generation_result_is_an_explicit_error(self):
        with self.assertRaisesRegex(
            run_sa.PipelineOutputError, "generation returned no output directory"
        ):
            run_sa._validate_generation_output(None, "gpt-oss-20b")

    def test_empty_atoms_are_not_silently_treated_as_abstention(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            atomize_dir = Path(temp_dir) / "atomize"
            atomize_dir.mkdir()
            (atomize_dir / "atoms.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(
                run_sa.PipelineOutputError, "atomization produced zero atoms"
            ):
                run_sa._validate_atoms_output(atomize_dir)

    def test_truncated_abstraction_confidence_is_rejected(self):
        source = [
            {
                "atom": "[A] [fact].",
                "index": 0,
                "abstractions": [
                    {"abstraction": "[A] [fact].", "index": 0}
                ],
            },
            {
                "atom": "[B] [fact].",
                "index": 1,
                "abstractions": [
                    {"abstraction": "[B] [fact].", "index": 0}
                ],
            },
        ]
        truncated = [
            {
                "atom": "[A] [fact].",
                "index": 0,
                "confidence": 80,
                "abstractions": [
                    {
                        "abstraction": "[A] [fact].",
                        "index": 0,
                        "confidence": 80,
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "abstractions_confidence.json"
            path.write_text(json.dumps(truncated), encoding="utf-8")
            with self.assertRaisesRegex(
                run_sa.PipelineOutputError,
                "abstraction-confidence count .* does not match abstraction count",
            ):
                run_sa._validate_abstraction_confidences(
                    path,
                    source,
                    {"[A] [fact].": 80, "[B] [fact].": 70},
                )

    def test_non_numeric_atom_confidence_is_rejected(self):
        atom = "[A] [fact]."
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "atoms_confidence.json"
            path.write_text(
                json.dumps({atom: {"confidence": None}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                run_sa.PipelineOutputError, "missing or outside"
            ):
                run_sa._validate_atom_confidences(path, [atom])


class OrchestrationTests(unittest.TestCase):
    def test_run_sa_real_stages_and_parsers_with_fake_client(self):
        responses = [
            "Entity was born in City.",
            "{'label': 'VALID', 'explanation': 'substantive synthetic answer'}",
            "1. [Entity] [was born in City].",
            (
                "1. [Entity] [was born in City].\n"
                "Reasoning: Original atomic fact.\n"
                "2. [Entity] [was born somewhere].\n"
                "Reasoning: Drop the city detail.\n"
                "3. STOP\n"
                "Reasoning: No further useful abstraction."
            ),
            (
                "1. Entity was born in City.\n"
                "Reasoning: Synthetic low confidence.\n"
                "Confidence: 20"
            ),
            (
                "1. Entity was born in City.\n"
                "Reasoning: Synthetic low confidence.\n"
                "Confidence: 20\n"
                "2. Entity was born somewhere.\n"
                "Reasoning: Synthetic high confidence.\n"
                "Confidence: 80"
            ),
            "Entity was born somewhere.",
        ]
        fake_client = _FakeClient(responses)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_root = Path(temp_dir)
            params_path = temp_root / "params.json"
            expected_params = {"temperature": 0.0, "top_p": 1.0}
            params_path.write_text(
                json.dumps({"gpt-oss-20b": expected_params}),
                encoding="utf-8",
            )
            output_root = temp_root / "outputs"
            normalized_output_root = Path(
                run_sa.normalize_output_root(output_root)
            )
            self.addCleanup(shutil.rmtree, str(normalized_output_root), True)

            with mock.patch.object(run_sa, "init_client", return_value=fake_client), \
                 mock.patch(
                     "sys.argv",
                     [
                         "run_sa.py",
                         "--input",
                         "Synthetic question",
                         "--output-dir",
                         str(output_root),
                         "--model",
                         "gpt-oss-20b",
                         "--theta",
                         "50",
                         "--generation-params-path",
                         str(params_path),
                         "--prompts-dir",
                         str(PIPELINE_DIR / "prompts"),
                     ],
                 ), \
                 mock.patch(
                     "socket.create_connection",
                     side_effect=AssertionError("Network access is forbidden in offline tests."),
                 ):
                run_sa.main()

            self.assertEqual(fake_client.chat.completions.responses, [])
            self.assertEqual(len(fake_client.chat.completions.calls), 7)
            for call in fake_client.chat.completions.calls:
                self.assertEqual(call["model"], "gpt-oss-20b")
                self.assertEqual(call["temperature"], 0.0)
                self.assertEqual(call["top_p"], 1.0)

            selected_paths = list(
                normalized_output_root.rglob("selected_statements.json")
            )
            self.assertEqual(len(selected_paths), 1)
            self.assertEqual(
                json.loads(selected_paths[0].read_text(encoding="utf-8")),
                ["[Entity] [was born somewhere]."],
            )
            reconstruction_paths = list(
                normalized_output_root.rglob("reconstruction/*/response.json")
            )
            self.assertEqual(len(reconstruction_paths), 1)
            reconstruction_data = json.loads(
                reconstruction_paths[0].read_text(encoding="utf-8")
            )
            self.assertEqual(
                reconstruction_data["choices"][0]["message"]["content"],
                "Entity was born somewhere.",
            )
            self.assertEqual(
                reconstruction_data["config"]["sys_prompt"], "recon_rf_v6"
            )

            self.assertEqual(len(list(normalized_output_root.rglob("atoms.json"))), 1)
            self.assertEqual(
                len(list(normalized_output_root.rglob("abstractions.json"))), 1
            )
            self.assertEqual(
                len(list(normalized_output_root.rglob("atoms_confidence.json"))), 1
            )
            self.assertEqual(
                len(
                    list(
                        normalized_output_root.rglob(
                            "abstractions_confidence.json"
                        )
                    )
                ),
                1,
            )

    def test_real_atomizer_empty_parse_fails_before_glue_guard(self):
        fake_client = _FakeClient(
            [
                "Entity was born in City.",
                "{'label': 'VALID', 'explanation': 'substantive synthetic answer'}",
                "No numbered atoms were returned.",
            ]
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_root = Path(temp_dir)
            params_path = temp_root / "params.json"
            params_path.write_text(
                json.dumps({"gpt-oss-20b": {"temperature": 0.0}}),
                encoding="utf-8",
            )
            output_root = temp_root / "outputs"
            normalized_output_root = run_sa.normalize_output_root(output_root)
            self.addCleanup(shutil.rmtree, normalized_output_root, True)

            with mock.patch.object(run_sa, "init_client", return_value=fake_client), \
                 mock.patch(
                     "sys.argv",
                     [
                         "run_sa.py",
                         "--input",
                         "Synthetic question",
                         "--output-dir",
                         str(output_root),
                         "--model",
                         "gpt-oss-20b",
                         "--theta",
                         "50",
                         "--generation-params-path",
                         str(params_path),
                         "--prompts-dir",
                         str(PIPELINE_DIR / "prompts"),
                     ],
                 ), \
                 mock.patch(
                     "socket.create_connection",
                     side_effect=AssertionError("Network access is forbidden in offline tests."),
                 ):
                # Historical non-chunked atomization divides by the parsed atom
                # count before run_sa can inspect atoms.json. Preserve and make
                # this failure explicit instead of claiming the glue catches it.
                with self.assertRaises(ZeroDivisionError):
                    run_sa.main()
            self.assertEqual(len(fake_client.chat.completions.calls), 3)

    def test_run_sa_stops_before_refusal_when_generation_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            params_path = temp_root / "params.json"
            params_path.write_text(
                json.dumps({"gpt-oss-20b": {"temperature": 1.0}}),
                encoding="utf-8",
            )
            with mock.patch.object(run_sa, "init_client", return_value=object()), \
                 mock.patch.object(run_sa, "run_generate_response", return_value=None), \
                 mock.patch.object(run_sa, "run_refusal_classification") as refusal_mock, \
                 mock.patch(
                     "sys.argv",
                     [
                         "run_sa.py",
                         "--input",
                         "Synthetic question",
                         "--output-dir",
                         str(temp_root / "outputs"),
                         "--model",
                         "gpt-oss-20b",
                         "--theta",
                         "50",
                         "--generation-params-path",
                         str(params_path),
                     ],
                 ), \
                 mock.patch(
                     "socket.create_connection",
                     side_effect=AssertionError("Network access is forbidden in offline tests."),
                 ):
                with self.assertRaisesRegex(
                    run_sa.PipelineOutputError,
                    "generation returned no output directory",
                ):
                    run_sa.main()
            refusal_mock.assert_not_called()

    def test_run_sa_keeps_model_params_and_v6_across_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            params_path = temp_root / "params.json"
            expected_params = {"temperature": 1.0, "top_p": 1.0}
            params_path.write_text(
                json.dumps({"gpt-oss-20b": expected_params}),
                encoding="utf-8",
            )
            output_root = temp_root / "outputs"
            fake_client = object()

            def fake_generate(**kwargs):
                generation_dir = Path(kwargs["input_dir"]) / "generate_response" / "prompt=longfact"
                generation_dir.mkdir(parents=True)
                (generation_dir / "response.json").write_text(
                    json.dumps(
                        {
                            "choices": [{"message": {"content": "Synthetic answer"}}],
                            "config": {"generate_prompt": "Synthetic question"},
                        }
                    ),
                    encoding="utf-8",
                )
                return str(generation_dir)

            def fake_atomize(**kwargs):
                atomize_dir = Path(kwargs["input_dir"]) / "atomize" / "synthetic"
                atomize_dir.mkdir(parents=True)
                (atomize_dir / "atoms.json").write_text(
                    json.dumps(["[Entity] [specific fact]."]),
                    encoding="utf-8",
                )
                return str(atomize_dir)

            def fake_abstract(**kwargs):
                path = Path(kwargs["atomize_output_dir"]) / "abstractions.json"
                path.write_text(
                    json.dumps(
                        [
                            {
                                "atom": "[Entity] [specific fact].",
                                "index": 0,
                                "abstractions": [
                                    {
                                        "abstraction": "[Entity] [specific fact].",
                                        "index": 0,
                                    },
                                    {
                                        "abstraction": "[Entity] [general fact].",
                                        "index": 1,
                                    },
                                ],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                return str(path)

            def fake_atom_confidence(**kwargs):
                path = Path(kwargs["input_files"]["atoms_file_path"]) / "atoms_confidence.json"
                path.write_text(
                    json.dumps({"[Entity] [specific fact].": {"confidence": 20}}),
                    encoding="utf-8",
                )
                return str(path)

            def fake_abstraction_confidence(**kwargs):
                path = temp_root / "abstractions_confidence.json"
                path.write_text(
                    json.dumps(
                        [
                            {
                                "atom": "[Entity] [specific fact].",
                                "index": 0,
                                "confidence": 20,
                                "abstractions": [
                                    {
                                        "abstraction": "[Entity] [specific fact].",
                                        "index": 0,
                                        "confidence": 99,
                                    },
                                    {
                                        "abstraction": "[Entity] [general fact].",
                                        "index": 1,
                                        "confidence": 50,
                                    },
                                ],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                return str(path)

            def fake_reconstruction(**kwargs):
                reconstruction_dir = Path(kwargs["input_dir"]) / "reconstruction" / "synthetic"
                reconstruction_dir.mkdir(parents=True)
                (reconstruction_dir / "response.json").write_text(
                    json.dumps(
                        {
                            "choices": [{"message": {"content": "General fact."}}],
                            "config": {
                                "generate_prompt": kwargs["original_prompt"],
                                "sys_prompt": kwargs["sys_prompt"],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return str(reconstruction_dir)

            def fake_refusal(**kwargs):
                response_path = Path(kwargs["input_dir"]) / "response.json"
                response = json.loads(response_path.read_text(encoding="utf-8"))
                response["is_refusal"] = False
                response["refusal_classification"] = {
                    "label": "VALID",
                    "explanation": "synthetic",
                }
                response_path.write_text(json.dumps(response), encoding="utf-8")
                return False

            with mock.patch.object(run_sa, "init_client", return_value=fake_client), \
                 mock.patch.object(run_sa, "run_generate_response", side_effect=fake_generate) as generate_mock, \
                 mock.patch.object(run_sa, "run_refusal_classification", side_effect=fake_refusal) as refusal_mock, \
                 mock.patch.object(run_sa, "run_atomize", side_effect=fake_atomize) as atomize_mock, \
                 mock.patch.object(run_sa, "run_abstract", side_effect=fake_abstract) as abstract_mock, \
                 mock.patch.object(run_sa, "run_atoms_confidence", side_effect=fake_atom_confidence) as atom_conf_mock, \
                 mock.patch.object(run_sa, "run_abstractions_confidence", side_effect=fake_abstraction_confidence) as abstraction_conf_mock, \
                 mock.patch.object(run_sa, "validate_confidences_for_entity_separate") as validate_mock, \
                 mock.patch.object(run_sa, "run_reconstruction", side_effect=fake_reconstruction) as reconstruction_mock, \
                 mock.patch(
                     "sys.argv",
                     [
                         "run_sa.py",
                         "--input",
                         "Synthetic question",
                         "--output-dir",
                         str(output_root),
                         "--model",
                         "gpt-oss-20b",
                         "--theta",
                         "50",
                         "--generation-params-path",
                         str(params_path),
                         "--prompts-dir",
                         str(PIPELINE_DIR / "prompts"),
                     ],
                 ), \
                 mock.patch(
                     "socket.create_connection",
                     side_effect=AssertionError("Network access is forbidden in offline tests."),
                 ):
                run_sa.main()

            for stage_mock in (
                generate_mock,
                refusal_mock,
                atomize_mock,
                abstract_mock,
                atom_conf_mock,
                abstraction_conf_mock,
                reconstruction_mock,
            ):
                self.assertIs(stage_mock.call_args.kwargs["client"], fake_client)
                self.assertEqual(stage_mock.call_args.kwargs["model"], "gpt-oss-20b")
                self.assertEqual(
                    stage_mock.call_args.kwargs["generation_params"],
                    expected_params,
                )

            reconstruction_args = reconstruction_mock.call_args.kwargs
            self.assertEqual(reconstruction_args["sys_prompt"], "recon_rf_v6")
            self.assertEqual(reconstruction_args["theta"], 50)
            self.assertEqual(
                reconstruction_args["statements"],
                ["[Entity] [general fact]."],
            )
            validate_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
