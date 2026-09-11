# TALLRec: Reverse-Engineered Product Requirements

## Product summary

TALLRec is a research toolkit that adapts a LLaMA 7B language model to binary recommendation. It converts a user's positive and negative interaction history into a natural-language instruction, fine-tunes small LoRA adapter matrices, and predicts whether the user will like a target movie or book (`Yes.` or `No.`).

The central value proposition is sample-efficient recommendation: researchers can reuse a general instruction-following LLM and tune only a small fraction of its parameters with as few as 16–256 domain examples. The intended users are recommender-systems and NLP researchers reproducing the TALLRec paper, comparing few-shot performance, or adapting its method to another catalog. This repository is an experiment pipeline, not an end-user recommendation application: it exposes no HTTP endpoints, persistent service, authentication system, or interactive UI.

## Personas and journeys

### Reproduction researcher

The researcher supplies a Hugging Face-format LLaMA checkpoint and the included movie or book JSON data. They start from instruction-tuned LoRA weights, run one or more sample-size/seed experiments, let validation AUC select the best checkpoint, evaluate the adapters on the test set, and compare the resulting AUC with the paper.

### ML/recommendation engineer

The engineer converts MovieLens 100K or Book-Crossing source data into instruction records, adjusts prompts or LoRA hyperparameters, trains a domain adapter, evaluates it, and optionally exports merged weights in Hugging Face or original LLaMA checkpoint format.

### Experiment operator

The operator edits the shell launchers with model, data, checkpoint, and output paths; chooses a GPU and seed; executes batches of experiments; and collects nested JSON metrics keyed by training domain, test domain, model name, seed, and sample count.

There is no consumer persona or direct recommendation journey in the code. A production product would need catalog ingestion, user/profile management, an inference API, authorization, monitoring, and a user-facing client.

## Functional requirements

### P0 — Core experiment path

1. **Load a LLaMA causal language model and tokenizer.** The system shall accept a local path or Hugging Face model identifier through `--base_model`.
2. **Tune with parameter-efficient LoRA.** It shall load the base model in 8-bit mode, prepare it for int8 training, and attach rank-configurable LoRA adapters to `q_proj` and `v_proj` by default.
3. **Train on recommendation instructions.** It shall load JSON datasets containing `instruction`, `input`, and `output`; render Alpaca-style prompts; tokenize them; and train with gradient accumulation and FP16.
4. **Support few-shot experiments.** It shall shuffle deterministically by seed and optionally select the first `sample` records.
5. **Validate on ranking quality.** It shall calculate ROC AUC from the LLaMA vocabulary probabilities associated with the expected `Yes.` and `No.` tokens, evaluate periodically, apply early stopping, retain one checkpoint, and restore the best model.
6. **Evaluate saved adapters.** It shall combine a base model with a LoRA adapter, generate batched responses for a test JSON file, compute ROC AUC, and append results to a JSON results hierarchy.
7. **Persist compact artifacts.** It shall save LoRA configuration and adapter weights instead of duplicating the full base model.

### P1 — Data and experiment support

8. **Provide two recommendation domains.** Checked-in train/validation/test instruction data shall cover MovieLens movies and Book-Crossing books.
9. **Generate movie prompts.** The preprocessing script shall create chronological ten-interaction histories, split the last 10,000 examples 80/10/10, and express ratings above 3 as `Yes.`.
10. **Generate book prompts.** The preprocessing script shall merge ratings with book metadata, keep users with more than three interactions, split users 80/10/10, and express ratings above 5 as `Yes.`.
11. **Support mixed-domain training.** A second training entry point shall sample and concatenate two training datasets for cross-domain experiments.
12. **Automate experiment matrices.** Shell scripts shall iterate over configured sample sizes, seeds, learning rates, dropout values, and adapter directories.

### P2 — Auxiliary capabilities

13. **General instruction tuning.** `finetune.py` shall support Alpaca-style data independent of the recommendation-specific AUC logic.
14. **Export merged models.** Export scripts shall merge a LoRA adapter into the base model and write either a Hugging Face checkpoint or an original LLaMA-style state dictionary.
15. **Optional experiment metadata.** Training shall accept Weights & Biases parameters, although recommendation training explicitly disables W&B reporting in the current implementation.

## Non-functional requirements inferred from the code

### Performance and hardware

- Training is optimized for a constrained research GPU through 8-bit base weights, FP16 compute, LoRA, gradient accumulation, length grouping, and optional multi-GPU/DDP behavior.
- The scripts target LLaMA 7B and assume CUDA for practical training. CPU and Apple MPS branches exist only in evaluation and will be much slower.
- Batched, left-padded inference improves evaluation throughput. The public `batch_size` argument is currently ignored: internal batches are always 32.
- `torch.compile` is enabled on non-Windows systems with PyTorch 2+, trading startup time for potential throughput improvement.

### Scalability and reliability

- Scaling is single-process/file-oriented. Hugging Face `Trainer` can participate in DDP when `WORLD_SIZE` and `LOCAL_RANK` are set, but there is no job queue, distributed data layer, model server, database, or concurrent result writer.
- Results are rewritten as one JSON document without locking or atomic replacement, so concurrent evaluation processes can lose updates or corrupt the file.
- Reproducibility is partial: training data is seeded, but complete deterministic PyTorch/CUDA settings and an environment lockfile are absent. Book preprocessing shuffles users without first setting a seed.
- Validation selects the best checkpoint by AUC and early stopping uses patience 10. Only one recommendation checkpoint is retained to constrain disk use.
- Input schemas are implicit and runtime validation is absent. The checked-in files are valid arrays of string-valued `instruction`, `input`, and `output` records.

### Security and privacy

- There is no network-facing attack surface in the repository itself and no credential handling or authentication.
- Model and dataset identifiers may cause Hugging Face downloads. Any gated LLaMA access token is handled outside this code by the Hugging Face client environment/cache.
- `preprocess_movie.py` calls Python `eval()` on CSV fields. A malicious or untrusted CSV can execute arbitrary code; safe parsing such as `ast.literal_eval()` is required before using external data.
- PyTorch `.bin` checkpoints are pickle-based and must be treated as executable/untrusted input. Only load weights from trusted sources.
- Raw recommender data contains user IDs and behavioral histories. The pipeline neither anonymizes them nor defines retention/access controls.

### Maintainability and compatibility

- The application is a collection of scripts with duplicated prompt/training logic, hard-coded token IDs, paths, and LLaMA 7B dimensions. There are no automated tests, CI workflow, structured configuration, packaging metadata, or error taxonomy.
- Dependencies are incompletely pinned. `torch`, `numpy`, `scikit-learn`, `pandas`, and `tqdm` are used but absent from `requirements.txt`; most listed packages have no version.
- Several defects prevent untouched end-to-end execution. These are catalogued in `ONBOARDING_GUIDE.md` so an operator can distinguish environment setup from code repair.

