# SPINACH-derived coverage components

The files listed below are derived from the official SPINACH project:
https://github.com/stanford-oval/spinach

SPINACH is distributed under the Apache License 2.0. The public release must
retain the upstream Apache-2.0 license and attribution alongside these files.
The copies here include local integration changes for the Selective Abstraction
coverage evaluator; this notice does not claim that they are byte-identical to
any particular upstream revision.

Derived Python files:

- `parser_python_interface.py`
- `wikidata_utils.py`
- `spinach_agent/parser_state.py`
- `spinach_agent/parser_utils.py`
- `spinach_agent/part_to_whole_parser.py`

Derived prompt files, copied without release-staging edits:

- `spinach_agent/prompts/controller.prompt`
- `spinach_agent/prompts/format_actions.prompt`
- `spinach_agent/prompts/conversation_reporter.prompt`
- `spinach_agent/prompts/language_detection.prompt`
- `spinach_agent/prompts/prune_outgoing_edges.prompt`
