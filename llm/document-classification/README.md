# document-classification

Classify and sort PDFs into topic folders using a **local** Qwen GGUF model via
[`llama.cpp`](https://github.com/ggerganov/llama.cpp). Point it at a folder of
PDFs and a list of topics; it reads the front matter of each document, asks the
model which topic fits best, and (optionally) copies the file into a matching
subfolder.

Everything runs on your machine — no documents are sent to any external service.

## How it works

1. **Read the front of each PDF.** A document's topic is almost always clear
   from its title, table of contents, preface, and first chapter, so only the
   first pages are extracted (capped at `MAX_PAGES` / `MAX_DOC_CHARS`). This keeps
   both PDF parsing and prompt processing fast without hurting accuracy.
2. **Prime the topics once.** The system prompt and full topics list (~10k
   tokens) are identical for every PDF, so they're evaluated a single time and
   the KV cache is snapshotted. Each document then only pays for its own ~2k
   tokens instead of re-processing the whole topics list every time.
3. **Classify deterministically.** The model is asked to print only the most
   specific matching topic (or `unsorted` if nothing fits). Reasoning is
   suppressed via an empty `<think>` block — for this task it cost 4–7× the time
   with no accuracy gain.
4. **Sort.** With `--copy-to`, each PDF is copied into `DEST/<topic>/`, where the
   topic path (e.g. `c++\concurrency`) becomes nested subfolders.

## Requirements

- Python **3.12+**
- A C/C++ build toolchain for `llama-cpp-python`:
  - **Windows:** Visual Studio Build Tools with the *Desktop development with C++* workload
  - **Linux/macOS:** a standard C/C++ compiler (gcc/clang)
- A GGUF model file (see below). GPU acceleration via CUDA is used when
  available (`n_gpu_layers=-1`); it also runs on CPU.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip in a virtual environment:

```bash
pip install -e .
```

> `llama-cpp-python` compiles native code on install. To build with CUDA
> support, set the appropriate build flags for your platform before installing,
> e.g. `CMAKE_ARGS="-DGGML_CUDA=on"` (see the
> [llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python#installation)).

## Getting a model

Download a Qwen GGUF model (quantized builds keep memory modest) and place it in
`models/`. Any `llama.cpp`-compatible GGUF works; the prompt template is tuned
for Qwen3. For example, a `Qwen3` instruct model in `Q4_K_M` quantization is a
good balance of speed and quality.

## The topics list

Topics live in a plain-text file named `topics`, one topic per line. Nested
topics use `\` (or `/`) as a path separator and become nested output folders:

```
android
android\flutter
android\kotlin
c++
c++\concurrency
c++\optimization
```

The model is asked to pick the **most specific** matching line.

## Usage

```bash
python classify_pdf.py <folder> --model models/<your-model>.gguf [--copy-to <dest>]
```

Examples:

```bash
# Just print the predicted topic for each PDF
python classify_pdf.py books-input --model models/Qwen3-27B-Q4_K_M.gguf

# Predict and copy each PDF into dest/<topic>/
python classify_pdf.py books-input --copy-to books-output --model models/Qwen3-27B-Q4_K_M.gguf
```

If the project is installed (`uv sync` / `pip install -e .`), a `classify-pdf`
command is available as an alias for `python classify_pdf.py`.

### Arguments

| Argument     | Required | Description                                                                 |
|--------------|----------|-----------------------------------------------------------------------------|
| `folder`     | yes      | Folder containing PDF files (searched recursively).                         |
| `--model`    | yes      | Path to the GGUF model file.                                                |
| `--copy-to`  | no       | If set, copy each PDF into `DEST/<topic>/`; unmatched files go to `unsorted`. |
| `--topics`   | no       | Path to the topics list, one topic per line (default: `topics`).            |

### Output

For each PDF the predicted topic is printed. If the input folder is organized so
that a PDF's parent folder is its true topic, the run also reports a running
match count and a final accuracy summary — handy for evaluating prompt or model
changes.

## Tuning

A few constants at the top of `classify_pdf.py` control extraction and context:

- `MAX_DOC_CHARS` — hard cap on document text sent to the model (default 14000).
- `MAX_PAGES` — never read more than this many pages (default 40).
- `N_CTX` — context window (default 32768).
