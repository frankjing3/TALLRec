# TALLRec: Project Execution and Onboarding Guide

## What actually starts the project

There is no single application bootstrap or development server. Choose an entry point based on the job:

| Job | Exact entry point |
|---|---|
| Primary TALLRec single-domain training | `finetune_rec.py` → `if __name__ == "__main__"` → `fire.Fire(train)` |
| Mixed movie/book training | `finetune_multi_rec.py` → `fire.Fire(train)` |
| Generic Alpaca instruction tuning | `finetune.py` → `fire.Fire(train)` |
| Test evaluation | `evaluate.py` → `fire.Fire(main)` |
| Data preparation | `preprocess_movie.py` or `preprocess_book.py` (top-level scripts) |
| Model export | `export_hf_checkpoint.py` or `export_state_dict_checkpoint.py` (top-level scripts) |

For the paper's core workflow, the absolute starting point is `finetune_rec.py`. `shell/instruct_7B.sh` is the intended launcher but contains placeholders and simply delegates to that file.

## Prerequisites

Use a Linux machine or Linux container. The pinned bitsandbytes 0.37.2 generation is Linux-oriented, the launchers are Bash, and training code sets a Linux library path. Native Windows is not a faithful environment for this snapshot.

Install:

- Git and Git LFS.
- Miniconda/Anaconda.
- An NVIDIA CUDA-capable GPU and compatible driver. LLaMA 7B training is impractical on CPU; 8-bit LoRA is intended to reduce, not eliminate, GPU requirements.
- A Hugging Face-format LLaMA 7B base checkpoint that you are licensed and authorized to use. The base model is not included.
- Optional Hugging Face authentication if the chosen model/dataset is gated.
- Raw MovieLens 100K or Book-Crossing files only if regenerating the checked-in data.

A conservative reproduction environment is Python 3.9 with PyTorch 1.13.1/CUDA 11.7. The repository pins Transformers 4.28.0, PEFT 0.3.0, and bitsandbytes 0.37.2 but does not pin Python or PyTorch. Treat this as a compatibility baseline, not a formally locked environment.

## Environment setup

From a Linux terminal:

```bash
cd /path/to/ThesisCode/TALLRec
git lfs install
git lfs pull

conda create -n tallrec python=3.9 -y
conda activate tallrec

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
  pytorch-cuda=11.7 -c pytorch -c nvidia

python -m pip install -r requirements.txt
python -m pip install numpy scikit-learn pandas tqdm
```

The second `pip` command supplies runtime imports omitted from `requirements.txt`. Verify the environment before loading a model:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import transformers, peft, bitsandbytes, datasets, fire, sklearn; print('imports OK')"
```

If the base model uses Hugging Face access controls:

```bash
huggingface-cli login
```

There is no required `.env` file. Configuration is supplied through CLI flags and shell environment variables. `BASE_MODEL` is required only by the two export scripts; `WORLD_SIZE` and `LOCAL_RANK` are read for distributed training; W&B variables are optional.

## Data format

Training, validation, and test files are JSON arrays:

```json
[
  {
    "instruction": "Given the user's preference ... answer Yes. or No.",
    "input": "User Preference: ...\nUser Unpreference: ...\nWhether ...?",
    "output": "Yes."
  }
]
```

Ready-to-use files are included under `data/movie/` and `data/book/`. Dataset sizes verified in this checkout are:

| Domain | Train | Validation | Test |
|---|---:|---:|---:|
| Movie | 8,000 | 1,000 | 1,000 |
| Book | 19,414 | 2,427 | 2,427 |

## Run a single-domain experiment

Set paths for your environment. Start with the included instruction adapter only when its `base_model_name_or_path` mismatch is understood; the adapter configuration contains an original author's local path, but PEFT evaluation uses the separately supplied base model.

```bash
export TALLREC_BASE_MODEL=/path/to/llama-7b-hf
export TALLREC_INIT_ADAPTER=./alpaca-lora-7B
export TALLREC_OUTPUT=./outputs/movie_seed0_64

CUDA_VISIBLE_DEVICES=0 python -u finetune_rec.py \
  --base_model "$TALLREC_BASE_MODEL" \
  --train_data_path ./data/movie/train.json \
  --val_data_path ./data/movie/valid.json \
  --output_dir "$TALLREC_OUTPUT" \
  --batch_size 128 \
  --micro_batch_size 32 \
  --num_epochs 200 \
  --learning_rate 1e-4 \
  --cutoff_len 512 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules '["q_proj","v_proj"]' \
  --train_on_inputs \
  --group_by_length \
  --resume_from_checkpoint "$TALLREC_INIT_ADAPTER" \
  --sample 64 \
  --seed 0
```

To train without initial instruction-tuned adapter weights, omit `--resume_from_checkpoint`. To train on all records, omit `--sample` or pass `--sample=-1`; however, the current code then leaves `eval_step` undefined, so the defect described below must first be fixed.

The shell launcher alternative is:

```bash
# First replace every XXX in shell/instruct_7B.sh with real paths.
bash ./shell/instruct_7B.sh 0 0
```

## Evaluate the adapter

The output directory name must end in `_SEED_SAMPLE`, because `evaluate.py` parses its final two underscore-separated segments. It also infers the training domain from whether the directory basename contains `book`.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --base_model "$TALLREC_BASE_MODEL" \
  --lora_weights ./outputs/movie_seed0_64 \
  --test_data_path ./data/movie/test.json \
  --result_json_data ./outputs/results.json \
  --load_8bit=True

cat ./outputs/results.json
```

The supplied batch evaluator can scan all matching adapter directories:

```bash
# Replace base_model and test_data placeholders in shell/evaluate.sh first.
bash ./shell/evaluate.sh 0 ./outputs/movie
```

No local server starts. Although Gradio is a dependency/import, no Gradio UI is constructed or launched.

## Optional preprocessing

MovieLens preprocessing expects `u.data`, `u.item`, and `u.user` in the current directory and writes intermediate files to `./data/`. Run it from a disposable working directory or adjust its paths so it does not overwrite the checked-in split files.

```bash
python preprocess_movie.py
```

Book preprocessing expects `BX-Book-Ratings.csv`, `BX-Users.csv`, and `BX-Books.csv` in the current directory:

```bash
python preprocess_book.py
```

Both scripts need repairs before use with current pandas/untrusted inputs; see the blockers below.

## Optional model export

The export scripts currently hard-code the adapter as `tloen/alpaca-lora-7b`, so edit that value if exporting a trained TALLRec adapter.

```bash
export BASE_MODEL=/path/to/llama-7b-hf
python export_hf_checkpoint.py
# output: ./hf_ckpt/

python export_state_dict_checkpoint.py
# output: ./ckpt/consolidated.00.pth and ./ckpt/params.json
```

## Known blockers and correctness risks

These are findings from the checked-out revision, not hypothetical production recommendations:

1. **Checkpoint load assignment is wrong.** In all training scripts, `model = set_peft_model_state_dict(...)` assigns `None` with PEFT 0.3.0. Call the function without assignment.
2. **Full-data recommendation training fails.** `eval_step` is assigned only when `sample > -1`; the default `sample=-1` leads to an unbound variable.
3. **Some computed step values are floats.** For samples above 128, `sample / 128 * 5` returns a float while Trainer step settings should be integers.
4. **Multi-domain launcher calls the wrong file.** `shell/instruct_multi_7B.sh` invokes `finetune_rec.py`, which does not accept the second dataset flags. It should invoke `finetune_multi_rec.py`.
5. **Multi-domain launcher uses a nonexistent shell variable.** It defines `val_data2` but passes `$val_data_path2`.
6. **Second validation dataset is discarded.** `finetune_multi_rec.py` loads `val_data2` but evaluates only `val_data`.
7. **Book preprocessing crashes.** `mx` is read before initialization. Current pandas also removed the `error_bad_lines` parameter.
8. **Movie preprocessing is unsafe for untrusted CSV.** It uses `eval()` to parse list fields.
9. **Evaluation ignores its batch-size flag.** The local batching helper defaults to 32 and is called without the CLI value.
10. **Evaluation depends on model-specific token IDs.** IDs 8241 and 3782 are assumed to represent the desired answer tokens. They must be verified for the exact tokenizer; changing the base model/tokenizer silently invalidates scoring.
11. **Evaluation metadata depends on directory names.** Adapter paths without at least two underscore-separated suffixes cause incorrect or failing seed/sample extraction.
12. **CPU evaluation can fail.** `model.half()` is called whenever `load_8bit` is false, including CPU, where FP16 operations may be unsupported.
13. **Hard-coded host path.** Training scripts overwrite `LD_LIBRARY_PATH` with an original developer's Conda directory. Remove that assignment and let the active environment configure its libraries.
14. **Environment is not reproducibly locked.** Several imported packages are absent from requirements and most versions are unconstrained.

The Python sources parse successfully and all six checked-in datasets conform to the expected three-string-field schema. Full training/evaluation was not executed because it requires the external LLaMA 7B checkpoint and a compatible legacy CUDA environment.

