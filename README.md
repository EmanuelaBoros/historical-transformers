# Historical Transformers

Research code and data-preparation material for experiments with transformer models on historical document tasks, with a focus on historical named entity recognition, archival question answering, and long-context language modeling.

The repository brings together three related work areas:

- **HIPE fine-tuning with LLaMA**: instruction-style fine-tuning and inference for French historical named entity recognition using HIPE 2020/2022 data.
- **ArchivalQA data preparation**: notebooks and resources for generating open-domain question answering data over archival news collections.
- **Long-horizon transformers**: experimental transformer and capsule-network components for longer document modeling.

## Repository Structure

```text
.
├── llama/
│   ├── data/hipe/                 # HIPE TSV and JSONL data used for NER fine-tuning
│   ├── data/prompts/              # Prompt templates for NER experiments
│   ├── ft_datasets/hipe_dataset.py # Custom HIPE instruction dataset wrapper
│   ├── finetuning.py              # LLaMA fine-tuning entry point
│   ├── inference.py               # Inference over the HIPE test split
│   ├── evaluate.py                # Converts generated tags back to HIPE-style TSV
│   └── HIPE-scorer/               # HIPE evaluation scripts
├── data_preparation/
│   └── archival-qa/               # ArchivalQA dataset generation framework
├── long-horizon-transformer/      # Long-context transformer and capsule experiments
└── README.md
```

## Main Use Cases

### 1. Fine-tune LLaMA for Historical NER

The `llama/` folder adapts Meta's LLaMA fine-tuning recipes to HIPE-style historical named entity recognition. The custom dataset wrapper reads the preprocessed files:

```text
llama/data/hipe/HIPE-2022-v2.1-hipe2020-train-fr_universal.jsonl
llama/data/hipe/HIPE-2022-v2.1-hipe2020-dev-fr_universal.jsonl
llama/data/hipe/HIPE-2022-v2.1-hipe2020-test-fr_universal.jsonl
```

These JSONL files contain sentence-level examples with:

| Field | Description |
|---|---|
| `tokens` | Input sentence tokens serialized as text |
| `tags` | Target output with inline entity tags |

The model is trained to generate tagged text from an instruction prompt.

### 2. Run Inference and Convert Predictions

After fine-tuning, `llama/inference.py` generates predictions for the HIPE test split. `llama/evaluate.py` then maps generated inline tags back into a HIPE-compatible TSV format for scoring.

### 3. Prepare ArchivalQA Data

The `data_preparation/archival-qa/` folder contains material from the ArchivalQA dataset generation framework. It includes notebooks for:

1. Article selection.
2. Question generation.
3. Syntactic and temporal processing.
4. General and temporal ambiguity filtering.
5. Triple-based filtering.

See [`data_preparation/archival-qa/README.md`](data_preparation/archival-qa/README.md) and [`data_preparation/archival-qa/Dataset_Generation_Framework/README.md`](data_preparation/archival-qa/Dataset_Generation_Framework/README.md) for details.

### 4. Explore Long-Horizon Transformer Experiments

The `long-horizon-transformer/` folder contains experimental code for masked language modeling and custom transformer/capsule architectures, including:

- `run_mlm.py` and `run_mlm_no_trainer.py` for masked language model training.
- `transformer.py`, `relative_transformer.py`, and `attention.py` for transformer components.
- Capsule-network implementations and experiments under `capsule/`, `CapsuleLayer/`, `CapsNet-Pytorch/`, and `CCCapsNet/`.

## Installation

Clone the repository:

```bash
git clone https://github.com/EmanuelaBoros/historical-transformers.git
cd historical-transformers
```

Create an environment for the LLaMA/HIPE experiments:

```bash
cd llama
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jsonlines nltk
```

The LLaMA experiments require access to the corresponding Hugging Face model weights, for example `meta-llama/Llama-2-7b-hf`.

## HIPE Data Preparation

The HIPE TSV files can be converted into the instruction-style JSONL format with:

```bash
cd llama/data/hipe
python preprocess.py
```

This creates files such as:

```text
HIPE-2022-v2.1-hipe2020-train-fr_universal.jsonl
HIPE-2022-v2.1-hipe2020-dev-fr_universal.jsonl
HIPE-2022-v2.1-hipe2020-test-fr_universal.jsonl
```

## Fine-Tuning

From the `llama/` directory, fine-tune LLaMA with LoRA and FSDP:

```bash
torchrun --nnodes 1 --nproc_per_node 2 finetuning.py \
  --enable_fsdp \
  --use_peft \
  --peft_method lora \
  --model_name meta-llama/Llama-2-7b-hf \
  --pure_bf16 \
  --output_dir ./exp/hipe_llama7b_prompt_1 \
  --use_fast_kernels \
  --dataset hipe_dataset \
  --save_model
```

Adjust `--nproc_per_node`, batch settings, and model path according to your GPU setup.









