# Byte-Level Byte Pair Encoding (BPE)

**A from-first-principles implementation of byte-level BPE training, developed while studying Stanford CS336: Language Modeling from Scratch.**

> **Release:** `v0.1.0` — reference / baseline implementation  
> **Scope:** BPE vocabulary and merge-rule training. The inference-time tokenizer and performance-optimized trainer are intentionally outside this initial release.

## Abstract

Tokenization is a critical interface between raw text and neural language models. Byte-level Byte Pair Encoding (BPE) builds a finite vocabulary of byte sequences by repeatedly merging frequently co-occurring adjacent tokens. Its behavior depends not only on the merge-selection rule but also on pre-tokenization, Unicode handling, special-token boundaries, and deterministic tie-breaking.

This repository implements the BPE **training** procedure in Python to make these design choices inspectable. The initial release prioritizes algorithmic clarity and reproducibility over throughput. A subsequent release will introduce a separately reviewable optimized implementation and compare it with this baseline under a controlled evaluation protocol.


## 1. Research and engineering objectives

The project investigates four questions:

1. **Correctness:** How can BPE merges be learned deterministically from UTF-8 text while preserving special-token boundaries?
2. **Representation:** How does the initial 256-symbol byte alphabet support arbitrary UTF-8 input without a character-level out-of-vocabulary problem?
3. **Complexity:** Which operations dominate the cost of a straightforward Python BPE trainer?
4. **Optimization:** Which improvements reduce wall-clock time or memory consumption **without changing the learned vocabulary and ordered merge rules**?

The first release establishes the reference behavior against which subsequent changes can be evaluated.

## 2. Algorithm

### 2.1 Input and output contract

The training entry point is expected to expose the following interface:

```python
train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```

- **Input:** a UTF-8 training corpus, target vocabulary size, and reserved special tokens.
- **Vocabulary:** a mapping from integer token IDs to raw byte sequences (`dict[int, bytes]`).
- **Merges:** an ordered sequence of byte-pair merges (`list[tuple[bytes, bytes]]`).

The ordered merge list is part of the learned model: its ordering determines the priority of merge operations during later encoding.

### 2.2 Training pipeline

```text
UTF-8 corpus
    |
    v
Split at special-token boundaries
    |
    v
GPT-2-style regex pre-tokenization
    |
    v
Count pre-token frequencies
    |
    v
Represent each pre-token as individual UTF-8 bytes
    |
    v
Count adjacent byte-token pairs (weighted by frequency)
    |
    v
Select the highest-priority pair and merge its occurrences
    |
    +---------------------- repeat ---------------------+
    |                                                   |
    v                                                   |
Update pair statistics and extend the vocabulary -------+
    |
    v
Return vocabulary and ordered merge rules
```

**Boundary invariant:** BPE merges are learned *within* pre-tokens. They must not cross a regex pre-token boundary or a reserved special-token boundary.

### 2.3 Byte-level initialization

The initial alphabet contains all 256 possible byte values:

$$
V_0 = \{\operatorname{bytes}([i]) \mid i \in \{0,\ldots,255\}\}.
$$

Reserved special tokens are assigned separate vocabulary entries. Their byte strings are not treated as ordinary merge candidates.

Unlike a character-level vocabulary, the byte-level alphabet can represent the UTF-8 encoding of any Unicode text. The resulting tokenization may still be inefficient for some languages or domains; coverage and compression efficiency are different properties.

### 2.4 Frequency-weighted pair selection

Let $w=(w_1,\ldots,w_{|w|})$ denote a pre-token represented as a sequence of current byte tokens, and let $f(w)$ be its frequency in the corpus. The count of an adjacent pair $(a,b)$ is

$$
C(a,b) = \sum_w f(w)\sum_{i=1}^{|w|-1}
\mathbf{1}[(w_i,w_{i+1})=(a,b)].
$$

At each iteration, select the pair with the greatest weighted frequency. If multiple pairs have equal frequency, the implementation must apply a consistent tie-breaking rule. For the Stanford CS336 specification, this is the **lexicographically greatest byte pair** among the tied candidates.

The selected pair $(a,b)$ is replaced by the concatenated byte token $a\Vert b$ at non-overlapping adjacent occurrences, and the new token is added to the vocabulary.

### 2.5 Stopping conditions

Training stops when the target vocabulary size is reached **or** when no further eligible adjacent pair exists. A requested vocabulary size does not guarantee that enough merge candidates exist in a small corpus.

## 3. Correctness invariants

A conforming implementation should preserve the following properties:

| Invariant | Expected behavior |
| --- | --- |
| Base alphabet | Exactly 256 single-byte tokens are available initially. |
| Special tokens | Reserved tokens are represented explicitly and excluded from ordinary merge training. |
| Pre-token boundaries | Merges never span two different regex pre-tokens. |
| Determinism | Identical input, configuration, and tie-breaking produce identical ordered merges. |
| Merge accounting | Pair counts include repeated occurrences and pre-token frequencies. |
| Vocabulary entries | Each learned token stores the concatenation of the merged byte sequences. |
| Output contract | The trainer returns `vocab` and `merges` in the documented types. |

These are **verification targets**, not claims that a particular test suite has already passed. Test results should be reported only after executing the tests on the released revision.

## 4. Repository scope — initial release

```text
.
├── README.md
├── bpe.py                  # Baseline BPE training implementation
├── pyproject.toml          # Python project and dependency configuration
├── uv.lock                 # Locked dependency resolution, if used
├── .python-version         # Python version preference, if used
└── .gitignore              # Excludes environments and generated artifacts
```

Only files actually needed to reproduce the baseline should be committed. If `bpe.py` imports a local helper, that helper must also be included. The optimized implementation, unfinished tokenizer, generated vocabulary files, profiler output, and local datasets are **not** part of `v0.1.0`.

## 5. Environment and reproducibility

### Requirements

- Python: follow the range specified in `pyproject.toml` (the accompanying CS336 starter configuration uses Python 3.12–3.13).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- A UTF-8 corpus that you have permission to use.

### Setup

```bash
# Clone the published repository.
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>

# Reproduce the environment from the project configuration.
uv sync
```

Replace the placeholder repository URL with the actual GitHub URL before publishing this README.

### Minimal API example

The following example assumes the public training function is named `train_bpe` and is defined in `bpe.py`:

```python
from pathlib import Path
from bpe import train_bpe

corpus_path = Path("demo_corpus.txt")
corpus_path.write_text(
    "low lower newest wider\nlow lower\n",
    encoding="utf-8",
)

vocab, merges = train_bpe(
    input_path=str(corpus_path),
    vocab_size=270,
    special_tokens=["<|endoftext|>"],
)

print(f"Vocabulary entries: {len(vocab)}")
print(f"Learned merges: {len(merges)}")
print("First merges:", merges[:10])
```

Run the example in the project environment. The precise number of learned merges depends on the corpus and available pair candidates. **Verify the function signature against the published `bpe.py` before releasing.**

## 6. Evaluation methodology

An optimized implementation is useful only if it preserves correctness. The evaluation protocol therefore separates **equivalence** from **performance**.

### 6.1 Functional validation

For each test corpus and vocabulary size:

1. Compare the learned `merges` lists in their exact order.
2. Compare vocabulary token IDs and byte sequences (or apply the equivalence criterion specified by the test).
3. Test special tokens, overlapping special-token names, Unicode, punctuation, repeated substrings, and empty or small corpora.
4. Repeat runs to check determinism.

When using the Stanford CS336 test suite, its adapters must be connected to this implementation before test results are meaningful.

### 6.2 Performance measurements

For each implementation, report:

- Corpus size (bytes), provenance, and SHA-256 digest.
- Target vocabulary size and number of special tokens.
- Python version, operating system, CPU model, and available RAM.
- Process count and other parallel-execution settings, if applicable.
- Wall-clock training time over repeated runs, including dispersion.
- Peak resident memory (RSS), measured consistently.
- Whether the outputs match the baseline exactly.

Use the same corpus and target vocabulary for all comparisons. Keep preprocessing, I/O, and startup costs consistently included or excluded; state which.

### 6.3 Results

No performance numbers are published in this baseline README because no reproducible measurements have yet been supplied for the exact released code.

| Implementation | Corpus | Target vocab | Time | Peak RSS | Equivalent output |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline (`v0.1.0`) | To be recorded | To be recorded | Not measured | Not measured | Reference |
| Optimized (future release) | Same corpus | Same target | Not measured | Not measured | To be verified |

**Planned speedup definition:**

$$
\mathrm{Speedup} = \frac{T_{\mathrm{baseline}}}{T_{\mathrm{optimized}}}.
$$

A speedup greater than 1 indicates lower measured wall-clock time for the optimized implementation under the stated conditions.

## 7. Computational considerations

A straightforward BPE trainer may repeatedly scan the tokenized corpus to recount adjacent pairs and apply merges. If $M$ is the number of learned merges and $S$ is the total number of token positions examined per iteration, the repeated-scan work is approximately $O(MS)$, excluding implementation-specific data-structure and sorting costs.

The practical bottlenecks may include:

- Unicode decoding and regex pre-tokenization.
- Python object allocation for byte-token sequences.
- Repeated adjacent-pair counting.
- Rewriting token sequences after each merge.
- Corpus memory residency and intermediate dictionaries.

These are potential costs to profile, **not measured bottlenecks claimed for this revision**.


## 8. Roadmap

- **`v0.1.0` — Baseline:** publish the reference BPE training implementation and document its behavior.
- **`v0.2.0` — Optimized trainer:** publish the multiprocessing-oriented implementation separately, document actual changes, and verify output equivalence.
- **Later:** complete the tokenizer (`encode`, `decode`, and streaming encode), add targeted tests, and investigate further performance improvements.

The optimized version will be introduced as a new Git commit and release tag, preserving the baseline for direct comparison.

## 9. References and attribution

- Stanford CS336, *Language Modeling from Scratch*: https://stanford-cs336.github.io/spring2025/

- Sennrich, Haddow, and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units*: https://aclanthology.org/P16-1162/
- GPT-2 (Radford et al., 2019): https://openai.com/research/better-language-models


---

*Engineering principle: establish correctness, make experiments reproducible, and measure before claiming optimization gains.*
