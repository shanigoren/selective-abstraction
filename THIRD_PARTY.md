# Third-party code and data

The root MIT license applies to original release code. The following upstream licenses remain in force for their respective components.

| Component | Use | License / source |
| --- | --- | --- |
| SPINACH | Adapted parser, Wikidata utilities, and controller prompts under `coverage/` | Apache-2.0; [upstream](https://github.com/stanford-oval/spinach), [license copy](LICENSES/SPINACH-Apache-2.0.txt), [adaptation notice](coverage/SPINACH_ORIGIN.md) |
| FActScore | External biography prompt inputs | MIT; [upstream](https://github.com/shmsw25/FActScore), [license copy](LICENSES/FActScore-MIT.txt) |
| LongFact-Objects | External factual prompt inputs | Apache-2.0 for `longfact/*`; [upstream](https://github.com/google-deepmind/long-form-factuality), [full upstream license notice](LICENSES/LongFact-Apache-2.0.txt) |

The LongFact upstream license file also includes the separate MIT notice for its vendored FActScore code. That notice does not change the Apache-2.0 designation for `longfact/*`.

The SPINACH adaptations support counting questions, the study's model configuration, and portable endpoint configuration. The upstream license is preserved. No SPINACH benchmark dataset is bundled.

Wikipedia and Wikidata are external knowledge services; their content and terms are governed by their providers. No retrieved content or count database is redistributed here. Models and Python dependencies are obtained separately under their own licenses and terms.

The DRAG and Chain-of-Verification implementations follow the baseline methods cited by the accompanying paper. Their names identify the methods; the repository does not include their source repositories or evaluation outputs.
