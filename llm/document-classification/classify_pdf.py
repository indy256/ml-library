"""Classify and sort PDFs into topic folders using a local Qwen GGUF model.

Usage:
    python classify_pdf.py books-input --model models/Qwen3-27B-Q4_K_M.gguf
    python classify_pdf.py books-input --copy-to books-output --model models/Qwen3-27B-Q4_K_M.gguf

See README.md for details.
"""

import argparse
import os
import shutil
import sys
import time

from pypdf import PdfReader
from llama_cpp import Llama

# A document's topic is determined by its front matter (title, table of
# contents, preface, first chapter). Feeding the whole book wastes time on
# both PDF text extraction and prompt processing without improving accuracy,
# so we read only the front of the document.
#
# MAX_DOC_CHARS  - hard cap on document text sent to the model.
# MAX_PAGES      - never read more than this many pages (bounds extraction
#                  time even for books whose early pages are text-sparse).
MAX_DOC_CHARS = 14000
MAX_PAGES = 40

# Context window. Comfortably fits the topics list (~10k tokens), the capped
# document (~5k tokens) and the model's reasoning + answer, while keeping the
# KV cache small. (Much smaller than the document length is large.)
N_CTX = 32768

# Default name of the file holding the topic list (one topic per line).
DEFAULT_TOPICS_FILE = "topics"


def extract_pdf_text(pdf_path: str) -> str | None:
    """Extract text from the front of the PDF.

    Reads pages until MAX_DOC_CHARS of text has been collected or MAX_PAGES
    have been scanned, whichever comes first.
    """
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        print(f"  SKIP (cannot open): {exc}")
        return None

    parts = []
    total = 0
    for i, page in enumerate(reader.pages):
        if i >= MAX_PAGES or total >= MAX_DOC_CHARS:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        if text.strip():
            parts.append(text)
            total += len(text)

    full_text = "\n".join(parts).strip()
    if not full_text:
        print("  SKIP (no extractable text)")
        return None
    return full_text[:MAX_DOC_CHARS]


def load_model(model_path: str) -> Llama:
    return Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=N_CTX,
        n_batch=512,
        verbose=False,
    )


# The prompt splits into a large STATIC part (system instructions + the full
# topics list) that is identical for every PDF, and a small per-document TAIL.
# The static part is ~10k tokens and re-processing it for every PDF dominated
# the runtime. We instead evaluate it once and snapshot the KV cache (see
# prime_topics); each document then only pays for its own ~2k tokens.
#
# Qwen3 also reasons by default, which here costs 4-7x the time with no accuracy
# gain and sometimes overruns the token cap into a truncated, garbage answer. We
# suppress reasoning by prefilling a closed, empty <think> block in the assistant
# turn (the documented Qwen3 way; the /no_think soft switch is ignored by this
# model). The completion is then just the topic line, so we drive the raw ChatML
# template directly instead of create_chat_completion.
SYSTEM_PROMPT = (
    "You are a precise document analyst. You are given a list of topics, one per line, and a document. Read the document and identify the most specific topic from the list this document belongs to."
)


def _prompt_parts(topics_path: str) -> tuple[str, str]:
    """Return the (static prefix, per-document suffix) of the prompt."""
    with open(topics_path, "r", encoding="utf-8") as file:
        topics = file.read()
    prefix = (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n"
        "Which of the following topics does the following document belong to. Choose most specific topic.\n\n"
        "=== TOPICS START ===\n"
        f"{topics}\n"
        "=== TOPICS END ===\n\n"
        "=== DOCUMENT START ===\n"
    )
    # Guidance placed right before the answer (measured to lift accuracy with no
    # performance cost): classify by subject rather than language/tool, and don't
    # over-specify. "Choose the deepest path" and few-shot examples were tried and
    # did not help (few-shot hurt).
    suffix = (
        "\n=== DOCUMENT END ===\n\n"
        "Print only the topic. Do not print anything else. If there is no match, print unsorted"
        "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return prefix, suffix


def prime_topics(llm: Llama, topics_path: str):
    """Evaluate the static prefix once and snapshot the KV cache.

    Returns (state, suffix) where `state` is the saved KV cache positioned at
    the end of the topics list, to be restored before each document.
    """
    prefix, suffix = _prompt_parts(topics_path)
    prefix_tokens = llm.tokenize(prefix.encode("utf-8"), add_bos=True, special=True)
    llm.reset()
    llm.eval(prefix_tokens)
    return llm.save_state(), suffix


def detect_topic(llm: Llama, state, suffix: str, document_text: str | None) -> str | None:
    if document_text is None:
        return None

    # Restore the cached topics KV, then evaluate only this document + suffix.
    llm.load_state(state)
    tail = llm.tokenize((document_text + suffix).encode("utf-8"), add_bos=False, special=True)

    out: list[int] = []
    for token in llm.generate(list(tail), temp=0.0, reset=False):
        out.append(token)
        text = llm.detokenize(out).decode("utf-8", errors="ignore")
        if token == llm.token_eos() or "\n" in text or "<|im_end|>" in text or len(out) > 64:
            break
    answer = llm.detokenize(out).decode("utf-8", errors="ignore").split("<|im_end|>")[0].strip()
    if not answer:
        return None
    return answer.splitlines()[0].strip()


def _norm_topic(topic: str | None) -> str:
    """Normalize a topic path for comparison (unify "\\" and "/" separators)."""
    return (topic or "").replace("\\", "/")


def copy_to_topic(pdf_path: str, dest_root: str, topic: str) -> str:
    """Copy `pdf_path` into dest_root/<topic>/, creating the subfolders.

    The topic is a path using "\\" or "/" separators; it becomes a nested
    subfolder under dest_root. Returns the destination directory.
    """
    parts = [p for p in topic.replace("\\", "/").split("/") if p]
    dest_dir = os.path.join(dest_root, *parts)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(pdf_path, os.path.join(dest_dir, os.path.basename(pdf_path)))
    return dest_dir


def find_pdfs(folder: str) -> list[str]:
    pdfs = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    pdfs.sort()
    return pdfs

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect topics of all PDFs in a folder using a Qwen GGUF model (CUDA)."
    )
    parser.add_argument("folder", help="Path to folder containing PDF files.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model file.",
    )
    parser.add_argument(
        "--copy-to",
        metavar="DEST",
        help="If set, copy each PDF into DEST under a subfolder named by the recognized topic.",
    )
    parser.add_argument(
        "--topics",
        default=DEFAULT_TOPICS_FILE,
        help=f"Path to the topics list, one topic per line (default: {DEFAULT_TOPICS_FILE}).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Not a directory: {args.folder}")
    if not os.path.isfile(args.topics):
        sys.exit(f"Topics file not found: {args.topics}")
    if not os.path.isfile(args.model):
        sys.exit(f"Model file not found: {args.model}")

    pdfs = find_pdfs(args.folder)
    if not pdfs:
        sys.exit(f"No PDF files found in {args.folder}")

    llm = load_model(args.model)
    state, suffix = prime_topics(llm, args.topics)

    matched = 0
    run_start = time.time()

    for i, pdf_path in enumerate(pdfs, 1):
        rel = os.path.relpath(pdf_path, args.folder)
        print(f"\n[{i}/{len(pdfs)}] {rel}")
        # When the input folder is organized by topic (a PDF's parent folder is
        # its true topic), we can score predictions. Otherwise actual_topic is
        # just "" and the match column is meaningless -- ignore it.
        actual_topic = os.path.dirname(rel)

        t0 = time.time()
        document_text = extract_pdf_text(pdf_path)

        predicted_topic = detect_topic(llm, state, suffix, document_text)
        match = _norm_topic(actual_topic) == _norm_topic(predicted_topic)
        if match:
            matched = matched + 1

        if args.copy_to:
            if predicted_topic:
                dest_dir = copy_to_topic(pdf_path, args.copy_to, predicted_topic)
            else:
                dest_dir = copy_to_topic(pdf_path, args.copy_to, 'unsorted')
            print(f"  copied -> {dest_dir}")

        elapsed = time.time() - t0
        print(f"Matches[{match} {matched}/{i}], predicted_topic {predicted_topic} for {pdf_path} ({elapsed:.1f}s)")

    total_time = time.time() - run_start
    print(f"\nMatched {matched}/{len(pdfs)} topics in {total_time:.1f}s")


if __name__ == "__main__":
    main()
