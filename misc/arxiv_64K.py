# ==================================================
# 0) Imports & Global Config
# ==================================================
import re, random, math, time, itertools, os
from collections import Counter
from typing import List
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from torch.nn.attention import sdpa_kernel, SDPBackend
import re
import random
from collections import Counter, defaultdict
from typing import List, Tuple
import numpy as np
from torch.nn.attention import sdpa_kernel, SDPBackend


# ==================================================
# 0) Config
# ==================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "ccdv/arxiv-classification"
TEXT_FIELD   = "text"
LABEL_FIELD  = "label"

# ---- Target sequence length and minimum raw length ----
PACK_TARGET_LEN    = 64_000   # final packed sequence length (context size)
MIN_DOC_LEN        = 1_000    # min raw length for individual docs used for packing
PACK_MIN_FRAC      = 0.8      # keep packed seqs >= 0.8 * PACK_TARGET_LEN

# How many docs you'd *like* total (before packing); actual may be smaller
DESIRED_TRAIN_TOTAL = 16000
DESIRED_TEST_TOTAL  = 6000
# ==================================================
# Retrieval-pair classification mode
# ==================================================
# "arxiv_classification" keeps your current task.
# "retrieval_pair" switches to LRA-style binary pair classification.
TASK_MODE = "retrieval_pair"   # "arxiv_classification" or "retrieval_pair"

# LRA Text Retrieval is commonly reported at 8K.
# Use 16K for a stronger long-context experiment.
RETRIEVAL_PAIR_LEN = 64_000     # set to 8_000 for exact @8000-style length
RETRIEVAL_TRAIN_PAIRS = 4000
RETRIEVAL_TEST_PAIRS  = 1000
RETRIEVAL_BATCH_SIZE  = 4       # reduce to 4 if memory is tight
RETRIEVAL_SEED = SEED + 777
TEXT_CONFIG = {
    "max_len": PACK_TARGET_LEN,  # model context length (pad/truncate to this)
    "vocab_limit": 50_000,
    "embed_dim": 256,
    "num_heads": 4,    # your choice
    "mlp_dim": 1024,
    "num_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    # LSH sparse attention params
    "lsh_num_bits": 5,            # 32 buckets
    "lsh_num_hashes": 2,          # average over 2 independent orderings
    "lsh_neighbor_buckets": 1,    # same bucket +/- 1
    "lsh_q_chunk_size": 1024,     # query chunk size for SDPA calls
    # RACE params
    "K": 4,
    "L": 4,
    "M": 1,
    # HyperAttention-style exact sparse branch
    "hyper_num_bits": 5,          # 32 buckets
    "hyper_block_size": 256,
    "hyper_min_seq_len": 4096,
    "hyper_neighbor_blocks": 0,   # start with 0 for max efficiency


    # Tiny gate MLP
    "gate_hidden_dim": 64,        # tiny 2-layer MLP hidden dim
    "gate_normalize": False,      # NSA-style independent sigmoid gates
    # Performer params
    "m_features": 256,
    "favor_seed": None,

    # training
    "batch_size": 4,
    "epochs": 50,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_accum_steps": 16,
        # --- hyper_race_mexact ---
    # m_exact = d_exact / (d_exact + lambda * d_race + eps)
    "mexact_eps": 1e-6,
    "mexact_lambda_learnable": True,
    "mexact_lambda_init": 1.0,

    # --- hyper_race_lambda_dependent ---
    # lambda_i = c + sigmoid(w^T q_i + b)
    "mexact_dependent_lambda_offset": 0.3,
    "mexact_dependent_lambda_offset_learnable": True,
    "mexact_dependent_lambda_offset_positive": True,
    "mexact_dependent_lambda_init_target": 0.8,
    "mexact_dependent_lambda_use_bias": True,
    "mexact_dependent_lambda_detach_q": True,
    "mexact_dependent_lambda_min": 1e-6,
    "mexact_dependent_lambda_w_init_std": 1e-3,
        # --- active task switch ---
    "task_mode": TASK_MODE,

    # --- LRA-style retrieval-pair classification ---
    "retrieval_pair_len": RETRIEVAL_PAIR_LEN,
    "retrieval_train_pairs": RETRIEVAL_TRAIN_PAIRS,
    "retrieval_test_pairs": RETRIEVAL_TEST_PAIRS,
    "retrieval_batch_size": RETRIEVAL_BATCH_SIZE,
    "retrieval_seed": RETRIEVAL_SEED,
}
if TASK_MODE == "retrieval_pair":
    TEXT_CONFIG["max_len"] = RETRIEVAL_PAIR_LEN
    TEXT_CONFIG["batch_size"] = RETRIEVAL_BATCH_SIZE
    TEXT_CONFIG["num_classes"] = 2
else:
    TEXT_CONFIG["max_len"] = PACK_TARGET_LEN
# ==================================================
# 1) Tokenizer (basic_english)
# ==================================================
_basic_english_re = re.compile(
    r"""([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])   # punctuation
     |(\d+[%]?)                                  # numbers (and percent)
     |([A-Za-z]+(?:'[A-Za-z]+)?)                 # words w/ optional apos
    """,
    re.VERBOSE,
)

def basic_english_tokenizer(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    for punc, num, word in _basic_english_re.findall(text):
        if punc:
            tokens.append(punc)
        elif num:
            tokens.append(num)
        elif word:
            tokens.append(word)
    return tokens

tok = basic_english_tokenizer

# ==================================================
# 2) (Optional) EDA augmenters
# ==================================================
def eda_random_deletion(tokens, p=0.05):
    if len(tokens) == 1:
        return tokens
    out = [t for t in tokens if random.random() > p]
    return out or [random.choice(tokens)]

def eda_random_swap(tokens, n_swaps=3):
    toks = tokens.copy()
    for _ in range(n_swaps):
        if len(toks) < 2:
            break
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return toks

# ==================================================
# 3) Helper: length stats
# ==================================================
def print_length_stats(arr: np.ndarray, name: str, thresholds=()):
    print("--------------------------------------------------")
    print(f"Raw token length stats ({name})")
    print("--------------------------------------------------")
    print(f"Count:      {len(arr)}")
    print(f"Min:        {int(arr.min())}")
    print(f"Max:        {int(arr.max())}")
    print(f"Mean:       {float(arr.mean()):.1f}")
    print(f"Median:     {int(np.median(arr))}")
    print(f"90th pct:   {int(np.percentile(arr, 90))}")
    print(f"95th pct:   {int(np.percentile(arr, 95))}")
    print(f"99th pct:   {int(np.percentile(arr, 99))}")
    for thr in thresholds:
        frac = float((arr >= thr).mean())
        print(f"Frac >= {thr:6d}: {frac:.3f}")
    print()

# ==================================================
# 4) Balanced subset with min-length constraint (per doc)
#    (IDENTICAL behavior to your "good" script)
# ==================================================
def make_balanced_long_examples(split, desired_total, min_len, name="train", seed=SEED):
    """
    Make a class-balanced subset where each *doc* has raw token length >= min_len.
    Returns:
      examples: list[(label, text)]
      num_classes: int
    """
    labels = list(split[LABEL_FIELD])
    texts  = list(split[TEXT_FIELD])

    print(f"\nBuilding balanced LONG-{name} subset with min_len = {min_len}...")
    print(f"Original {name} split size: {len(labels)}")

    # Precompute lengths
    print(f"Tokenizing {name} split to compute lengths...")
    lengths = []
    for txt in texts:
        toks = tok(str(txt))
        lengths.append(len(toks))
    lengths = np.array(lengths, dtype=np.int32)

    # Bucket by class, keeping only long docs
    buckets = defaultdict(list)
    for idx, (y, L) in enumerate(zip(labels, lengths)):
        y_int = int(y)
        if L >= min_len:
            buckets[y_int].append(idx)

    num_classes = len(buckets)
    if num_classes == 0:
        raise ValueError(f"No examples meet min_len = {min_len} in {name} split!")

    print(f"Found {num_classes} classes with at least one example >= min_len.")
    for y in sorted(buckets.keys()):
        print(f"  Class {y}: {len(buckets[y])} examples >= {min_len}")

    # Compute per-class quota
    max_possible_per_class = min(len(idxs) for idxs in buckets.values())
    desired_per_class      = desired_total // num_classes
    per_class              = min(max_possible_per_class, desired_per_class)

    if per_class == 0:
        raise ValueError(
            f"min_len = {min_len} is too strict: at least one class has 0 long examples."
        )

    actual_total = per_class * num_classes
    print(f"\nDesired total {name} examples: {desired_total}")
    print(f"Max possible per class (given min_len): {max_possible_per_class}")
    print(f"Using per_class = {per_class}, so actual total = {actual_total}")

    # Sample per class
    rng = random.Random(seed)
    chosen_idx = []
    for y, idxs in buckets.items():
        rng.shuffle(idxs)
        chosen_idx.extend(idxs[:per_class])
    rng.shuffle(chosen_idx)

    examples = [(int(labels[i]), texts[i]) for i in chosen_idx]

    # Stats for the final subset of docs
    final_lengths = lengths[chosen_idx]
    print_length_stats(
        final_lengths,
        f"{name} docs (balanced, length-filtered)",
        thresholds=(min_len,),
    )

    return examples, num_classes

# ==================================================
# 5) Streaming packer: use all tokens up to 64k chunks
#    (IDENTICAL behavior to your "good" script)
# ==================================================
def pack_examples_streaming(
    examples,
    target_len=PACK_TARGET_LEN,
    min_frac=PACK_MIN_FRAC,
    seed=SEED,
):
    """
    Streaming packer that:
      - groups docs by label
      - iterates through docs per class, tokenizing and appending into a buffer
      - emits a packed example every time buffer hits target_len
      - emits a final partial example if it's >= min_frac * target_len

    This reuses residual tokens from long docs rather than discarding them.
    """
    rng = random.Random(seed)
    per_class_docs = defaultdict(list)

    for lbl, txt in examples:
        per_class_docs[int(lbl)].append(str(txt))

    new_examples = []

    for lbl, docs in per_class_docs.items():
        # Shuffle docs within class to randomize packing
        rng.shuffle(docs)

        cur_tokens = []
        for txt in docs:
            toks = tok(txt)
            j = 0
            n = len(toks)
            while j < n:
                remaining_space = target_len - len(cur_tokens)
                if remaining_space <= 0:
                    # Buffer full → emit
                    if len(cur_tokens) >= int(min_frac * target_len):
                        new_examples.append((lbl, " ".join(cur_tokens)))
                    cur_tokens = []
                    remaining_space = target_len

                take = min(remaining_space, n - j)
                if take <= 0:
                    break

                cur_tokens.extend(toks[j : j + take])
                j += take

                if len(cur_tokens) == target_len:
                    # Emit full packed sequence
                    new_examples.append((lbl, " ".join(cur_tokens)))
                    cur_tokens = []

        # End of docs for this class: flush leftover if big enough
        if len(cur_tokens) >= int(min_frac * target_len):
            new_examples.append((lbl, " ".join(cur_tokens)))
        # else: drop tiny tail

    return new_examples

# ==================================================
# 6) Dataset class (same as "good" script)
# ==================================================
class ArxivDataset(Dataset):
    def __init__(self, examples, max_len, stoi, pad_idx=0, unk_idx=1, augment=False):
        self.examples = examples
        self.max_len  = max_len
        self.augment  = augment
        self.stoi     = stoi
        self.pad_idx  = pad_idx
        self.unk_idx  = unk_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        lbl, txt = self.examples[idx]
        toks = tok(str(txt))

        if self.augment:
            op = random.choice(["del", "swap", None])
            if op == "del":
                toks = eda_random_deletion(toks)
            elif op == "swap":
                toks = eda_random_swap(toks)

        toks = toks[: self.max_len]
        ids  = [self.stoi.get(t, self.unk_idx) for t in toks]
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        return int(lbl), torch.tensor(ids, dtype=torch.long)

    def collate_fn(self, batch):
        labels, tokens = zip(*batch)
        tokens = torch.stack(tokens, dim=0)
        masks  = (tokens != self.pad_idx).long()
        return tokens, masks, torch.tensor(labels, dtype=torch.long)

# ==================================================
# 7) Effective length stats from DataLoader masks
# ==================================================
def compute_effective_lengths_from_loader(dl, num_batches=100):
    all_lengths = []
    for i, (tokens, masks, labels) in enumerate(dl):
        lens = masks.sum(dim=1).cpu().numpy()
        all_lengths.append(lens)
        if i + 1 >= num_batches:
            break
    if not all_lengths:
        return np.array([], dtype=np.int32)
    return np.concatenate(all_lengths).astype(np.int32)

def print_effective_length_stats(arr: np.ndarray, max_len: int, thresholds=()):
    print("--------------------------------------------------")
    print("Effective sequence length stats (after padding/truncation)")
    print("--------------------------------------------------")
    print(f"Count (sampled): {len(arr)}")
    if len(arr) == 0:
        print("No data collected from DataLoader.")
        return
    print(f"Min:             {int(arr.min())}")
    print(f"Max:             {int(arr.max())}  (max_len = {max_len})")
    print(f"Mean:            {float(arr.mean()):.1f}")
    print(f"Median:          {int(np.median(arr))}")
    for thr in thresholds:
        frac = float((arr >= thr).mean())
        print(f"Frac >= {thr:6d}: {frac:.3f}")
    print()

print("Loading dataset:", DATASET_NAME)
raw = load_dataset(DATASET_NAME)

if "validation" in raw:
    train_split = raw["train"]
    test_split  = raw["validation"]
elif "test" in raw:
    train_split = raw["train"]
    test_split  = raw["test"]
else:
    tmp = raw["train"].train_test_split(test_size=0.2, seed=SEED)
    train_split, test_split = tmp["train"], tmp["test"]

# -----------------------------------------------
# 8.1) Build balanced, long-doc subsets (>= MIN_DOC_LEN)
# -----------------------------------------------
train_docs, num_classes_train = make_balanced_long_examples(
    train_split,
    desired_total=DESIRED_TRAIN_TOTAL,
    min_len=MIN_DOC_LEN,
    name="train",
    seed=SEED,
)
test_docs, num_classes_test = make_balanced_long_examples(
    test_split,
    desired_total=DESIRED_TEST_TOTAL,
    min_len=MIN_DOC_LEN,
    name="test",
    seed=SEED,
)

assert num_classes_train == num_classes_test
num_classes = num_classes_train

print(f"Final balanced-long train docs (>= {MIN_DOC_LEN}): {len(train_docs)}")
print(f"Final balanced-long test docs  (>= {MIN_DOC_LEN}): {len(test_docs)}")
print(f"Num classes: {num_classes}\n")

# -----------------------------------------------
# 8.2) STREAMING: pack docs into ~64k sequences per class
# -----------------------------------------------
print(f"Streaming-pack long docs into ~{PACK_TARGET_LEN} token sequences...")
train_examples_packed = pack_examples_streaming(
    train_docs,
    target_len=PACK_TARGET_LEN,
    min_frac=PACK_MIN_FRAC,
    seed=SEED,
)
test_examples_packed = pack_examples_streaming(
    test_docs,
    target_len=PACK_TARGET_LEN,
    min_frac=PACK_MIN_FRAC,
    seed=SEED + 1,
)

print(f"Packed train size (~{PACK_TARGET_LEN}): {len(train_examples_packed)}")
print(f"Packed test size  (~{PACK_TARGET_LEN}): {len(test_examples_packed)}\n")

# Stats on packed sequences (raw token counts)
def packed_lengths(examples):
    return np.array([len(tok(str(txt))) for _, txt in examples], dtype=np.int32)

train_packed_lengths = packed_lengths(train_examples_packed)
test_packed_lengths  = packed_lengths(test_examples_packed)

print_length_stats(
    train_packed_lengths,
    name="train_packed (~64k)",
    thresholds=(int(PACK_MIN_FRAC * PACK_TARGET_LEN), PACK_TARGET_LEN),
)
print_length_stats(
    test_packed_lengths,
    name="test_packed (~64k)",
    thresholds=(int(PACK_MIN_FRAC * PACK_TARGET_LEN), PACK_TARGET_LEN),
)

# Use packed examples from here on
train_examples = train_examples_packed
test_examples  = test_examples_packed

# -----------------------------------------------
# 8.3) Build vocab from packed train examples
# -----------------------------------------------
print("Building vocabulary from packed train examples...")
counter = Counter()
for lbl, txt in train_examples:
    counter.update(tok(str(txt)))

most_common = [w for w, _ in counter.most_common(TEXT_CONFIG["vocab_limit"])]
stoi = {w: i + 2 for i, w in enumerate(most_common)}
stoi["<pad>"] = 0
stoi["<unk>"] = 1

PAD_IDX, UNK_IDX = 0, 1
VOCAB_SIZE = len(stoi)
TEXT_CONFIG["vocab_size"]  = VOCAB_SIZE
TEXT_CONFIG["num_classes"] = num_classes

print(f"Vocab size: {VOCAB_SIZE}\n")

# -----------------------------------------------
# 8.4) Create datasets / loaders at 64k
# -----------------------------------------------
max_len  = TEXT_CONFIG["max_len"]  # 64_000
batch_sz = TEXT_CONFIG["batch_size"]  

train_ds = ArxivDataset(
    train_examples,
    max_len=max_len,
    stoi=stoi,
    pad_idx=PAD_IDX,
    unk_idx=UNK_IDX,
    augment=True,
)
test_ds = ArxivDataset(
    test_examples,
    max_len=max_len,
    stoi=stoi,
    pad_idx=PAD_IDX,
    unk_idx=UNK_IDX,
    augment=False,
)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_sz,
    shuffle=True,
    drop_last=True,
    pin_memory=(DEVICE == "cuda"),
    num_workers=4,
    collate_fn=train_ds.collate_fn,
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_sz,
    shuffle=False,
    pin_memory=(DEVICE == "cuda"),
    num_workers=2,
    collate_fn=test_ds.collate_fn,
)

print(f"Train batches: {len(train_dl)}")
print(f"Test  batches: {len(test_dl)}\n")

# -----------------------------------------------
# 8.5) Effective length stats from DataLoader
# -----------------------------------------------
eff_lengths = compute_effective_lengths_from_loader(
    train_dl,
    num_batches=100,  # sample
)
print_effective_length_stats(
    eff_lengths,
    max_len=max_len,
    thresholds=(int(PACK_MIN_FRAC * max_len), max_len),
)

print("Done. Packed sequences are ~62k tokens long with minimal padding.")
# ==================================================
# 8.6) Optional LRA-style retrieval-pair classification
#
# This does NOT remove the previous arXiv classification pipeline.
# It simply overrides train_dl/test_dl when:
#
#     TEXT_CONFIG["task_mode"] == "retrieval_pair"
#
# Task:
#   input = [CLS] document_a [SEP] document_b
#   label = 1 if same arXiv class, 0 otherwise
#
# This is an easy LRA-style binary text retrieval benchmark at 8K/16K.
# ==================================================

def make_lra_style_retrieval_pairs(
    source_examples,
    num_pairs: int,
    seed: int = SEED,
):
    """
    Build balanced binary retrieval-pair examples from already-packed
    class-labeled documents.

    source_examples:
        list[(class_label, text)]

    Returns:
        list[(pair_label, text_a, text_b)]

    pair_label:
        1 = same class
        0 = different class
    """
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for y, txt in source_examples:
        by_label[int(y)].append(str(txt))

    all_labels = [y for y, docs in by_label.items() if len(docs) >= 1]
    pos_labels = [y for y, docs in by_label.items() if len(docs) >= 2]

    if len(all_labels) < 2:
        raise ValueError("Need at least two classes for negative retrieval pairs.")

    if len(pos_labels) == 0:
        raise ValueError("Need at least one class with >=2 packed examples for positive pairs.")

    n_pos = num_pairs // 2
    n_neg = num_pairs - n_pos

    pairs = []

    # Positive pairs: two documents from the same class.
    for _ in range(n_pos):
        y = rng.choice(pos_labels)
        a, b = rng.sample(by_label[y], 2)
        pairs.append((1, a, b))

    # Negative pairs: two documents from different classes.
    for _ in range(n_neg):
        y1, y2 = rng.sample(all_labels, 2)
        a = rng.choice(by_label[y1])
        b = rng.choice(by_label[y2])
        pairs.append((0, a, b))

    rng.shuffle(pairs)
    return pairs


class RetrievalPairDataset(Dataset):
    """
    Binary retrieval-pair classification dataset.

    Each example is encoded as:

        [CLS] text_a[:left_budget] [SEP] text_b[:right_budget]

    The final sequence is padded/truncated to max_len.
    """

    def __init__(
        self,
        pair_examples,
        max_len,
        stoi,
        pad_idx=0,
        unk_idx=1,
        cls_idx=None,
        sep_idx=None,
    ):
        self.examples = pair_examples
        self.max_len = int(max_len)
        self.stoi = stoi
        self.pad_idx = int(pad_idx)
        self.unk_idx = int(unk_idx)

        if cls_idx is None:
            raise ValueError("cls_idx must be provided.")
        if sep_idx is None:
            raise ValueError("sep_idx must be provided.")

        self.cls_idx = int(cls_idx)
        self.sep_idx = int(sep_idx)

        # Reserve two positions: one for CLS and one for SEP.
        usable = self.max_len - 2
        if usable <= 0:
            raise ValueError("max_len must be >= 3.")

        self.left_budget = usable // 2
        self.right_budget = usable - self.left_budget

    def __len__(self):
        return len(self.examples)

    def _encode_tokens(self, toks, budget):
        toks = toks[:budget]
        return [self.stoi.get(t, self.unk_idx) for t in toks]

    def __getitem__(self, idx):
        pair_label, txt_a, txt_b = self.examples[idx]

        toks_a = tok(str(txt_a))
        toks_b = tok(str(txt_b))

        ids_a = self._encode_tokens(toks_a, self.left_budget)
        ids_b = self._encode_tokens(toks_b, self.right_budget)

        ids = [self.cls_idx] + ids_a + [self.sep_idx] + ids_b

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]

        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))

        return int(pair_label), torch.tensor(ids, dtype=torch.long)

    def collate_fn(self, batch):
        labels, tokens = zip(*batch)
        tokens = torch.stack(tokens, dim=0)
        masks = (tokens != self.pad_idx).long()
        labels = torch.tensor(labels, dtype=torch.long)
        return tokens, masks, labels


if TEXT_CONFIG.get("task_mode", "arxiv_classification") == "retrieval_pair":
    print("\n" + "=" * 90)
    print("Switching active task to LRA-style arXiv retrieval-pair classification")
    print("=" * 90)

    # Update active model/task configuration.
    TEXT_CONFIG["max_len"] = int(TEXT_CONFIG.get("retrieval_pair_len", 16_000))
    TEXT_CONFIG["num_classes"] = 2
    TEXT_CONFIG["batch_size"] = int(TEXT_CONFIG.get("retrieval_batch_size", 8))

    # Add special tokens used by the pair-classification input format.
    if "<cls>" not in stoi:
        stoi["<cls>"] = len(stoi)
    if "<sep>" not in stoi:
        stoi["<sep>"] = len(stoi)

    CLS_IDX = stoi["<cls>"]
    SEP_IDX = stoi["<sep>"]

    VOCAB_SIZE = len(stoi)
    TEXT_CONFIG["vocab_size"] = VOCAB_SIZE

    # Use the already-packed arXiv examples as long documents.
    # These were created above by pack_examples_streaming(...).
    retrieval_train_pairs = make_lra_style_retrieval_pairs(
        train_examples_packed,
        num_pairs=int(TEXT_CONFIG.get("retrieval_train_pairs", 4000)),
        seed=int(TEXT_CONFIG.get("retrieval_seed", SEED + 777)),
    )

    retrieval_test_pairs = make_lra_style_retrieval_pairs(
        test_examples_packed,
        num_pairs=int(TEXT_CONFIG.get("retrieval_test_pairs", 1000)),
        seed=int(TEXT_CONFIG.get("retrieval_seed", SEED + 777)) + 1,
    )

    print(f"Retrieval train pairs: {len(retrieval_train_pairs)}")
    print(f"Retrieval test pairs : {len(retrieval_test_pairs)}")
    print(f"Retrieval max_len    : {TEXT_CONFIG['max_len']}")
    print(f"Pair format          : [CLS] doc_a [SEP] doc_b")
    print(f"Label 1              : same arXiv class")
    print(f"Label 0              : different arXiv classes")
    print(f"Vocab size           : {TEXT_CONFIG['vocab_size']}")
    print(f"CLS_IDX              : {CLS_IDX}")
    print(f"SEP_IDX              : {SEP_IDX}")

    train_ds = RetrievalPairDataset(
        retrieval_train_pairs,
        max_len=TEXT_CONFIG["max_len"],
        stoi=stoi,
        pad_idx=PAD_IDX,
        unk_idx=UNK_IDX,
        cls_idx=CLS_IDX,
        sep_idx=SEP_IDX,
    )

    test_ds = RetrievalPairDataset(
        retrieval_test_pairs,
        max_len=TEXT_CONFIG["max_len"],
        stoi=stoi,
        pad_idx=PAD_IDX,
        unk_idx=UNK_IDX,
        cls_idx=CLS_IDX,
        sep_idx=SEP_IDX,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=TEXT_CONFIG["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=(DEVICE == "cuda"),
        num_workers=4,
        collate_fn=train_ds.collate_fn,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=TEXT_CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=(DEVICE == "cuda"),
        num_workers=2,
        collate_fn=test_ds.collate_fn,
    )

    print(f"Retrieval train batches: {len(train_dl)}")
    print(f"Retrieval test batches : {len(test_dl)}")

    eff_lengths = compute_effective_lengths_from_loader(
        train_dl,
        num_batches=50,
    )
    print_effective_length_stats(
        eff_lengths,
        max_len=TEXT_CONFIG["max_len"],
        thresholds=(TEXT_CONFIG["max_len"],),
    )

    print("Active task is now retrieval_pair.")
    print("=" * 90 + "\n")

# ==================================================
# 5) Attention modules (all baselines from vision)
#     – text version is pad-mask aware
# ==================================================
def gray_code_corners(num_bits: int, device):
    """
    Return the 2^num_bits hypercube corners in Gray-code order.
    Adjacent bucket IDs differ by one bit, which makes neighboring
    bucket indices more geometrically meaningful.
    Output shape: [R, num_bits]
    """
    R = 1 << num_bits
    corners = []
    for i in range(R):
        g = i ^ (i >> 1)  # Gray code
        bits = []
        for b in range(num_bits - 1, -1, -1):
            bit = (g >> b) & 1
            bits.append(1.0 if bit == 1 else -1.0)
        corners.append(bits)
    return torch.tensor(corners, dtype=torch.float32, device=device)
def _gray_code_order(num_bits: int, device):
    """
    Gray-code order so adjacent bucket IDs differ by one bit.
    Returns a LongTensor of length 2^num_bits.
    """
    if num_bits == 1:
        return torch.tensor([0, 1], device=device, dtype=torch.long)

    def rec(n):
        if n == 1:
            return torch.tensor([0, 1], device=device, dtype=torch.long)
        a = rec(n - 1)
        return torch.cat([a, torch.flip(a, dims=[0]) + (1 << (n - 1))], dim=0)

    return rec(num_bits)


def _gather_tokens_3d(x: torch.Tensor, idx: torch.Tensor):
    """
    x   : [H, T, D]
    idx : [H, S]
    returns gathered x along token dim -> [H, S, D]
    """
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def _run_exact_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Q, K, V : [B', H', L, D]
    Returns    : [B', H', L, D]

    Uses FlashAttention-backed SDPA on CUDA if possible.
    """
    if Q.device.type == "cuda":
        Q16, K16, V16 = [t.to(dtype=torch.float16) for t in (Q, K, V)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                Q16, K16, V16,
                dropout_p=0.0,
                is_causal=False,
            )
        return out.to(Q.dtype)
    else:
        return F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=0.0,
            is_causal=False,
        )
class ExactFlashAttention(nn.Module):
    """
    Exact full attention using SDPA / FlashAttention backend when available.

    This is the text-version analogue of the exact_flash attention from vit.py,
    but made pad-mask aware.

    IMPORTANT:
    - If there is no padding in the batch, it runs one full batched SDPA call.
    - If padding exists, it slices each sample to its valid length and runs exact
      full attention per sample, so PAD tokens are truly excluded.
    """
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        """
        x   : [B, T, d]
        mask: [B, T] with 1 for real tokens, 0 for PAD
        """
        B, T, _ = x.shape
        H, D = self.h, self.dk

        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()

        # Fast path: no padding at all -> one exact batched SDPA call
        if mask is None or bool((mask.sum(dim=1) == T).all().item()):
            out = _run_exact_sdpa(Q, K, V)  # [B,H,T,D]
        else:
            # Pad-aware path: slice each sample to valid_T so PAD tokens are excluded
            out = torch.zeros_like(Q)  # [B,H,T,D]
            for b in range(B):
                valid_T = int(mask[b].sum().item())
                if valid_T == 0:
                    continue

                Qh = Q[b:b+1, :, :valid_T, :]   # [1,H,Tv,D]
                Kh = K[b:b+1, :, :valid_T, :]
                Vh = V[b:b+1, :, :valid_T, :]

                out[b:b+1, :, :valid_T, :] = _run_exact_sdpa(Qh, Kh, Vh)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)  # [B,T,d]
        out = self.drop(out)
        out = self.o(out)

        # keep PAD positions clean
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        return out

class MultiHeadAttention(nn.Module):
    """Standard softmax MH attention with pad mask, using SDPA."""
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        """
        x:    [B, T, d]
        mask: [B, T] with 1 for real tokens, 0 for PAD
        """
        B, T, _ = x.shape
        h, dk = self.h, self.dk

        # [B, T, d] -> [B, H, T, D]
        Q = self.q(x).view(B, T, h, dk).transpose(1, 2)
        K = self.k(x).view(B, T, h, dk).transpose(1, 2)
        V = self.v(x).view(B, T, h, dk).transpose(1, 2)

        Q, K, V = [t.to(dtype=torch.float16) for t in (Q, K, V)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.0,      # we keep dropout on the output like before
                is_causal=False,
            )
        # [B, H, T, D] -> [B, T, H*D]
        out = out.transpose(1, 2).contiguous().view(B, T, h * dk)
        out = self.drop(out)
        out = out.to(self.o.weight.dtype)
        return self.o(out)
class AngularLSHGray(nn.Module):
    """
    HyperAttention-style hard angular LSH with Gray-code bucket ordering.
    Input expected in shape [..., T, D].
    Output is integer bucket IDs in shape [..., T].
    """
    def __init__(self, num_bits: int, dim: int, device="cpu"):
        super().__init__()
        self.num_bits = num_bits
        self.R = 1 << num_bits

        proj_dir = torch.randn(dim, num_bits, device=device)
        perm = _gray_code_order(num_bits, device=device)
        enc_vec = (2 ** torch.arange(num_bits, device=device, dtype=torch.long)).view(
            *([1] * 2), num_bits
        )

        self.register_buffer("proj_dir", proj_dir, persistent=False)   # [D, num_bits]
        self.register_buffer("perm", perm, persistent=False)           # [R]
        self.register_buffer("enc_vec", enc_vec, persistent=False)     # [1,1,num_bits]

    def hash(self, mat: torch.Tensor):
        """
        mat: [H, T, D] or [B, H, T, D]
        return: [H, T] or [B, H, T]
        """
        # project onto random hyperplanes
        proj = torch.einsum("...td,dr->...tr", mat, self.proj_dir)  # [..., T, num_bits]
        bits = (proj > 0).to(torch.long)
        bin_ids = (bits * self.enc_vec).sum(dim=-1)                 # [..., T]
        return self.perm[bin_ids]                                   # Gray-ordered IDs
class AngularAttention(nn.Module):
    """Angular (cosine) attention."""
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        B, T, _ = x.shape
        h, dk = self.h, self.dk

        Q = self.q(x).view(B, T, h, dk).transpose(1, 2)
        K = self.k(x).view(B, T, h, dk).transpose(1, 2)
        V = self.v(x).view(B, T, h, dk).transpose(1, 2)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        sim = (Q @ K.transpose(-2, -1)).clamp(-0.999, 0.999)
        scores = 1.0 - torch.acos(sim) / math.pi
        if mask is not None:
            pad = mask[:, None, None, :]
            scores = scores.masked_fill(pad == 0, 0.0)

        W = scores.clamp(min=1e-6).pow(8)
        W = W / (W.sum(-1, keepdim=True) + 1e-6)
        W = self.drop(W)

        out = (W @ V).transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.o(out)

class BatchedACE(nn.Module):
    """Non-causal ACE used inside RACE, adapted from vision."""
    def __init__(self, d_k, K, L, M, device="cpu", share_planes=False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer("planes_T", planes.view(L * K, d_k).T)
        else:
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)
            self.register_buffer("planes_T", planes)

        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)), device=device)
        self.register_buffer("protos_T", corners.T)

        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, Khf, Vhf, Qhf, eps=1e-6):
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        if self.share_planes:
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            projK = Kh2 @ self.planes_T
            projQ = Qh2 @ self.planes_T
        else:
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            projK = torch.einsum("mbtd,mds->mbts", Kh2, self.planes_T)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, self.planes_T)

            projK = projK.contiguous().view(M * BH, T, self.L * self.K)
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2    = V2.view(M * BH, T, dk)
            N     = M * BH

        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        logitsK = (projK.tanh().div(scale) @ self.protos_T)   # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)
        probsK  = F.softmax(logitsK, dim=-1)
        probsQ  = F.softmax(logitsQ, dim=-1)

        probsK_S = probsK.contiguous().view(N, T, S)
        probsQ_S = probsQ.contiguous().view(N, T, S)

        b_sum = probsK_S.transpose(1, 2).bmm(V2)      # [N,S,dk]
        A     = probsK_S.sum(dim=1)                   # [N,S]
        E     = b_sum / (A.unsqueeze(-1) + eps)       # [N,S,dk]

        out2 = probsQ_S.bmm(E)                        # [N,T,dk]
        out  = out2.view(M, B, H, T, dk).permute(0, 1, 2, 3, 4)
        return out

class RACEAttention(nn.Module):
    def __init__(self, d, h, drop, K, L, M, qkv_bias=False, device="cpu"):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.M = h, d // h, M
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.dk, K, L, M, device=device)

    def forward(self, x, mask):
        B, T, d = x.shape
        h, dk, M = self.h, self.dk, self.M

        Q = self.q(x).view(B, T, h, dk)
        K = self.k(x).view(B, T, h, dk)
        V = self.v(x).view(B, T, h, dk)

        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1).to(Q.dtype)
            Q, K, V = Q * m, K * m, V * m

        def pack(z):
            return z.unsqueeze(0).expand(M, -1, -1, -1, -1)

        out_m = self.ace(pack(K), pack(V), pack(Q))   # [M,B,H,T,dk]
        out   = out_m.mean(dim=0)                     # [B,H,T,dk]
        out   = out.permute(0, 2, 1, 3).contiguous().view(B, T, h * dk)
        return self.drop(self.o(out))
# New attention module
class HyperLSHExactAttention(nn.Module):
    """
    HyperAttention-style exact sparse attention:

    - hard angular LSH
    - sort queries and keys by bucket ID
    - reorder values with keys
    - compute dense exact attention on aligned blocks
    - inverse-permute query outputs back to original order

    This implements the exact block-diagonal idea from HyperAttention
    in the style of arxiv_64K.py using SDPA/FlashAttention.
    """

    def __init__(
        self,
        d,
        h,
        drop,
        num_bits=5,          # 32 buckets by default
        block_size=256,
        min_seq_len=4096,
        neighbor_blocks=0,   # 0 = same block only (fastest)
        qkv_bias=False,
        device="cpu",
    ):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.num_bits = num_bits
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.dk, device=device)

    def _full_sdpa_fallback(self, Qh, Kh, Vh):
        """
        Qh,Kh,Vh: [H, T, D]
        Returns  : [H, T, D]
        """
        out = _run_exact_sdpa(
            Qh.unsqueeze(0),  # [1,H,T,D]
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]
        return out

    def _same_block_exact(self, Qs, Ks, Vs, valid_T):
        """
        Fast path: same-block exact attention only.
        Qs,Ks,Vs: [H, T_valid, D] already sorted.
        Returns : [H, T_valid, D]
        """
        H, T_valid, D = Qs.shape
        bsz = self.block_size

        num_full_blocks = T_valid // bsz
        rem = T_valid % bsz

        out_sorted = torch.zeros_like(Qs)

        # Process full blocks in one big batched SDPA call
        if num_full_blocks > 0:
            T_full = num_full_blocks * bsz

            Q_full = Qs[:, :T_full, :].view(H, num_full_blocks, bsz, D)
            K_full = Ks[:, :T_full, :].view(H, num_full_blocks, bsz, D)
            V_full = Vs[:, :T_full, :].view(H, num_full_blocks, bsz, D)

            # Flatten (H, num_blocks) into batch dimension, use H'=1
            Q_flat = Q_full.reshape(H * num_full_blocks, 1, bsz, D)
            K_flat = K_full.reshape(H * num_full_blocks, 1, bsz, D)
            V_flat = V_full.reshape(H * num_full_blocks, 1, bsz, D)

            O_flat = _run_exact_sdpa(Q_flat, K_flat, V_flat)  # [H*num_blocks,1,bsz,D]
            O_full = O_flat.reshape(H, num_full_blocks, bsz, D).reshape(H, T_full, D)
            out_sorted[:, :T_full, :] = O_full

        # Process final partial block (if any)
        if rem > 0:
            q_last = Qs[:, num_full_blocks * bsz:, :]   # [H, rem, D]
            k_last = Ks[:, num_full_blocks * bsz:, :]
            v_last = Vs[:, num_full_blocks * bsz:, :]

            o_last = _run_exact_sdpa(
                q_last.unsqueeze(0),   # [1,H,rem,D]
                k_last.unsqueeze(0),
                v_last.unsqueeze(0),
            )[0]
            out_sorted[:, num_full_blocks * bsz:, :] = o_last

        return out_sorted

    def _neighbor_block_exact(self, Qs, Ks, Vs, valid_T):
        """
        Slower but richer path: each sorted query block attends to
        same block plus neighboring blocks in sorted key order.
        Qs,Ks,Vs: [H, T_valid, D] already sorted.
        Returns : [H, T_valid, D]
        """
        H, T_valid, D = Qs.shape
        bsz = self.block_size
        num_blocks = math.ceil(T_valid / bsz)

        out_sorted = torch.zeros_like(Qs)

        for bi in range(num_blocks):
            q0 = bi * bsz
            q1 = min((bi + 1) * bsz, T_valid)

            left = max(0, bi - self.neighbor_blocks)
            right = min(num_blocks - 1, bi + self.neighbor_blocks)

            k0 = left * bsz
            k1 = min((right + 1) * bsz, T_valid)

            q_blk = Qs[:, q0:q1, :]   # [H, q_len, D]
            k_blk = Ks[:, k0:k1, :]
            v_blk = Vs[:, k0:k1, :]

            o_blk = _run_exact_sdpa(
                q_blk.unsqueeze(0),    # [1,H,q_len,D]
                k_blk.unsqueeze(0),    # [1,H,k_len,D]
                v_blk.unsqueeze(0),
            )[0]
            out_sorted[:, q0:q1, :] = o_blk

        return out_sorted

    def forward(self, x, mask=None):
        """
        x   : [B, T, d]
        mask: [B, T] with 1 for real tokens, 0 for PAD
        """
        B, T, _ = x.shape
        H, D = self.h, self.dk

        # Standard projections
        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()

        # Zero padded tokens so fallback path won't blow up with junk
        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q = Q * keep
            K = K * keep
            V = V * keep

        out = torch.zeros_like(Q)  # [B,H,T,D]

        # Per-sample processing, because each sample can have different valid_T
        # and its own independent sorting permutations.
        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue

            Qh = Q[b, :, :valid_T, :]   # [H,Tv,D]
            Kh = K[b, :, :valid_T, :]
            Vh = V[b, :, :valid_T, :]

            # Short-sequence fallback: just run exact full attention
            if valid_T < self.min_seq_len:
                out[b, :, :valid_T, :] = self._full_sdpa_fallback(Qh, Kh, Vh)
                continue

            # HyperAttention-style hard LSH sorting per head
            q_bucket_ids = self.lsh.hash(Qh)   # [H,Tv]
            k_bucket_ids = self.lsh.hash(Kh)   # [H,Tv]

            q_sort_idx = torch.argsort(q_bucket_ids, dim=1, stable=True)
            k_sort_idx = torch.argsort(k_bucket_ids, dim=1, stable=True)
            q_sort_inv = torch.argsort(q_sort_idx, dim=1, stable=True)

            Qs = _gather_tokens_3d(Qh, q_sort_idx)  # [H,Tv,D]
            Ks = _gather_tokens_3d(Kh, k_sort_idx)
            Vs = _gather_tokens_3d(Vh, k_sort_idx)

            if self.neighbor_blocks == 0:
                O_sorted = self._same_block_exact(Qs, Ks, Vs, valid_T)
            else:
                O_sorted = self._neighbor_block_exact(Qs, Ks, Vs, valid_T)

            # Inverse-permute query outputs back to original order
            O_unsorted = O_sorted.gather(
                1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D)
            )
            out[b, :, :valid_T, :] = O_unsorted

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)  # [B,T,d]
        out = self.drop(out)
        return self.o(out)
class LSHBucketSparseAttention(nn.Module):
    """
    GPU-friendlier sparse attention prototype:

    - hard-assign queries/keys to the most probable bucket
    - Gray-ordered buckets so neighboring bucket IDs differ by 1 bit
    - sort Q by query buckets, sort K/V by key buckets
    - for each query bucket, attend only to same + neighboring buckets
    - compute dense SDPA/FlashAttention on those contiguous slices
    - inverse-permute query outputs back to original order
    - average outputs across multiple independent hash rounds

    IMPORTANT:
    This is a prototype written in the style of arxiv_64K.py.
    It is much more GPU-friendly than arbitrary token top-k retrieval,
    but it is still not as optimized as a custom fused block-sparse kernel.
    """

    def __init__(
        self,
        d,
        h,
        drop,
        num_bits=5,             # 2^5 = 32 buckets by default
        num_hashes=2,           # average over 2 independent hash rounds
        neighbor_buckets=1,     # same bucket +/- 1 neighbors
        q_chunk_size=1024,      # chunk sorted queries for memory safety
        qkv_bias=False,
        device="cpu",
    ):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.num_bits = num_bits
        self.R = 1 << num_bits
        self.num_hashes = num_hashes
        self.neighbor_buckets = neighbor_buckets
        self.q_chunk_size = q_chunk_size

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        # One independent Gaussian hyperplane set per hash round
        planes = torch.randn(num_hashes, self.dk, num_bits, device=device)
        self.register_buffer("planes_T", planes)   # [num_hashes, dk, num_bits]

        # Gray-ordered bucket prototypes (corners of {-1,+1}^num_bits)
        corners = gray_code_corners(num_bits, device=device)  # [R, num_bits]
        self.register_buffer("protos_T", corners.T.contiguous())  # [num_bits, R]

        # Learnable softness, same spirit as current RACE code
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0, device=device)))

    def _bucket_probs_and_ids(self, Xh: torch.Tensor, hash_idx: int):
        """
        Xh: [H, T, dk] for one sample
        Returns:
            probs_head : [H, T, R]
            probs_avg  : [T, R]   (mean over heads, used for one shared sorting order)
            bucket_ids : [T]      hard bucket assignment = argmax over probs_avg
        """
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        # [H,T,dk] x [dk,num_bits] -> [H,T,num_bits]
        proj = torch.einsum("htd,dp->htp", Xh, self.planes_T[hash_idx])

        # Compare against Gray-ordered corner prototypes
        logits = (proj.tanh().div(scale) @ self.protos_T)  # [H,T,R]
        probs_head = F.softmax(logits, dim=-1)             # [H,T,R]

        # Shared token order across heads: average bucket probs over heads
        probs_avg = probs_head.mean(dim=0)                 # [T,R]
        bucket_ids = probs_avg.argmax(dim=-1)              # [T]

        return probs_head, probs_avg, bucket_ids

    def _sort_by_bucket(self, Xh: torch.Tensor, bucket_ids: torch.Tensor):
        """
        Xh: [H, T, dk]
        bucket_ids: [T]
        Returns sorted tensor and metadata.
        """
        perm = torch.argsort(bucket_ids, stable=True)      # [T]
        Xh_sorted = Xh[:, perm, :]                         # [H,T,dk]
        ids_sorted = bucket_ids[perm]                      # [T]

        counts = torch.bincount(ids_sorted, minlength=self.R)  # [R]
        starts = torch.zeros_like(counts)
        if counts.numel() > 1:
            starts[1:] = torch.cumsum(counts[:-1], dim=0)
        ends = starts + counts

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

        return Xh_sorted, ids_sorted, perm, inv_perm, starts, ends

    def _dense_sdpa(self, Qh: torch.Tensor, Kh: torch.Tensor, Vh: torch.Tensor):
        """
        Qh: [H, q_len, dk]
        Kh: [H, k_len, dk]
        Vh: [H, k_len, dk]
        Returns:
            [H, q_len, dk]
        """
        Q = Qh.unsqueeze(0)  # [1,H,q_len,dk]
        K = Kh.unsqueeze(0)  # [1,H,k_len,dk]
        V = Vh.unsqueeze(0)  # [1,H,k_len,dk]

        if Q.device.type == "cuda":
            # Use FlashAttention backend if available through SDPA
            Q16, K16, V16 = [t.to(dtype=torch.float16) for t in (Q, K, V)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    Q16, K16, V16,
                    dropout_p=0.0,
                    is_causal=False,
                )
            out = out.to(Q.dtype)
        else:
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.0,
                is_causal=False,
            )

        return out[0]  # [H,q_len,dk]

    def _single_hash_sparse_attention(self, Qh, Kh, Vh, hash_idx):
        """
        One sparse attention realization for one sample and one hash round.
        Inputs:
            Qh,Kh,Vh: [H, T_valid, dk]
        Returns:
            out: [H, T_valid, dk]
        """
        # Hard bucket assignments for queries and keys
        _, _, q_bucket_ids = self._bucket_probs_and_ids(Qh, hash_idx)
        _, _, k_bucket_ids = self._bucket_probs_and_ids(Kh, hash_idx)

        # Sort queries by query buckets, keys/values by key buckets
        Qs, _, _, inv_q_perm, q_starts, q_ends = self._sort_by_bucket(Qh, q_bucket_ids)
        Ks, _, _, _,           k_starts, k_ends = self._sort_by_bucket(Kh, k_bucket_ids)
        Vs, _, _, _,           _,        _      = self._sort_by_bucket(Vh, k_bucket_ids)

        H, T_valid, dk = Qs.shape
        out_sorted = torch.zeros_like(Qs)

        # For each query bucket, attend to same bucket +/- neighbors
        for r in range(self.R):
            q0 = int(q_starts[r].item())
            q1 = int(q_ends[r].item())
            if q1 <= q0:
                continue

            left_bucket  = max(0, r - self.neighbor_buckets)
            right_bucket = min(self.R - 1, r + self.neighbor_buckets)

            k0 = int(k_starts[left_bucket].item())
            k1 = int(k_ends[right_bucket].item())
            if k1 <= k0:
                continue

            K_slice = Ks[:, k0:k1, :]   # [H, k_len, dk]
            V_slice = Vs[:, k0:k1, :]

            # Chunk queries inside this bucket for memory safety
            for qs in range(q0, q1, self.q_chunk_size):
                qe = min(qs + self.q_chunk_size, q1)
                Q_slice = Qs[:, qs:qe, :]  # [H, q_len, dk]
                out_slice = self._dense_sdpa(Q_slice, K_slice, V_slice)
                out_sorted[:, qs:qe, :] = out_slice

        # Inverse-permute query outputs back to original order
        out = out_sorted[:, inv_q_perm, :]   # [H, T_valid, dk]
        return out

    def forward(self, x, mask=None):
        """
        x   : [B, T, d]
        mask: [B, T], 1 for real tokens, 0 for PAD
        """
        B, T, _ = x.shape
        H, dk = self.h, self.dk

        # Standard projections
        Q = self.q(x).view(B, T, H, dk).transpose(1, 2).contiguous()  # [B,H,T,dk]
        K = self.k(x).view(B, T, H, dk).transpose(1, 2).contiguous()  # [B,H,T,dk]
        V = self.v(x).view(B, T, H, dk).transpose(1, 2).contiguous()  # [B,H,T,dk]

        # Mask padded tokens by zeroing; valid-length slicing below avoids
        # sorting/attending on pads.
        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)  # [B,1,T,1]
            Q = Q * keep
            K = K * keep
            V = V * keep

        out = torch.zeros_like(Q)  # [B,H,T,dk]

        # NOTE:
        # We loop over batch items because each sample can have different valid length
        # and different per-sample bucket segment boundaries.
        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue

            Qh = Q[b, :, :valid_T, :]  # [H,T_valid,dk]
            Kh = K[b, :, :valid_T, :]
            Vh = V[b, :, :valid_T, :]

            out_accum = torch.zeros_like(Qh)

            # IMPORTANT:
            # We AVERAGE outputs across hash rounds.
            # We do NOT majority-vote bucket IDs, because bucket labels from
            # independent random hash rounds are not semantically aligned.
            for hash_idx in range(self.num_hashes):
                out_accum += self._single_hash_sparse_attention(Qh, Kh, Vh, hash_idx)

            out_accum = out_accum / float(self.num_hashes)
            out[b, :, :valid_T, :] = out_accum

        # Merge heads, dropout, output projection
        out = out.transpose(1, 2).contiguous().view(B, T, H * dk)  # [B,T,d]
        out = self.drop(out)
        return self.o(out)
#New Task
def race_bucket_probs_from_qk(attn: RACEAttention, Q: torch.Tensor, K: torch.Tensor):
    """
    Reproduce the bucket-probability part of BatchedACE using the SAME
    random planes / prototypes / temperature as the existing RACEAttention.

    Inputs
    ------
    Q, K : [B, T, H, dk]

    Returns
    -------
    probsQ, probsK : [M, B, H, T, L, R]
    """
    ace = attn.ace
    M = attn.M

    B, T, H, dk = Q.shape

    def pack(z):
        return z.unsqueeze(0).expand(M, -1, -1, -1, -1)   # [M,B,T,H,dk]

    Qhf = pack(Q)
    Khf = pack(K)

    M_, B_, T_, H_, dk_ = Khf.shape
    scale = ace.logit_temp.exp().clamp(1e-2, 20.0)

    if ace.share_planes:
        # Collapse M*B*H -> N
        N = M_ * B_ * H_
        Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T_, dk_)  # [N,T,dk]
        Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T_, dk_)  # [N,T,dk]

        projK = Kh2 @ ace.planes_T    # [N,T,L*K]
        projQ = Qh2 @ ace.planes_T    # [N,T,L*K]
    else:
        BH = B_ * H_

        Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M_, BH, T_, dk_)  # [M,BH,T,dk]
        Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M_, BH, T_, dk_)  # [M,BH,T,dk]

        projK = torch.einsum("mbtd,mds->mbts", Kh2, ace.planes_T)  # [M,BH,T,L*K]
        projQ = torch.einsum("mbtd,mds->mbts", Qh2, ace.planes_T)  # [M,BH,T,L*K]

        projK = projK.contiguous().view(M_ * BH, T_, ace.L * ace.K)
        projQ = projQ.contiguous().view(M_ * BH, T_, ace.L * ace.K)

        N = M_ * BH

    projK = projK.view(N, T_, ace.L, ace.K)   # [N,T,L,K]
    projQ = projQ.view(N, T_, ace.L, ace.K)   # [N,T,L,K]

    logitsK = (projK.tanh().div(scale) @ ace.protos_T)   # [N,T,L,R]
    logitsQ = (projQ.tanh().div(scale) @ ace.protos_T)   # [N,T,L,R]

    probsK = F.softmax(logitsK, dim=-1)                  # [N,T,L,R]
    probsQ = F.softmax(logitsQ, dim=-1)                  # [N,T,L,R]

    # reshape back to [M,B,H,T,L,R]
    probsK = probsK.view(M_, B_, H_, T_, ace.L, ace.R)
    probsQ = probsQ.view(M_, B_, H_, T_, ace.L, ace.R)

    return probsQ, probsK 

#Again New Task
class HyperRaceGatedAttention(nn.Module):
    """
    Hybrid attention:
      1) Hyper-LSH exact sparse attention branch
      2) RACE attention branch
      3) 2-layer tiny MLP gate that outputs 2 scalar gates per token
      4) weighted sum of the two branch outputs

    IMPORTANT:
    - This assumes HyperLSHExactAttention is already defined in your file.
    - This also uses the existing RACEAttention already in your file.
    - The gate is query-side: it uses the current token representation x
      (in practice, x will already be normalized by the block before calling att()).
    """

    def __init__(self, cfg, device=DEVICE):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]

        # ---- Branch 1: HyperAttention-style exact sparse attention ----
        self.hyper = HyperLSHExactAttention(
            d=d,
            h=h,
            drop=drop,
            num_bits=cfg.get("hyper_num_bits", 5),              # default: 32 buckets
            block_size=cfg.get("hyper_block_size", 256),
            min_seq_len=cfg.get("hyper_min_seq_len", 4096),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
        )

        # ---- Branch 2: RACE attention ----
        self.race = RACEAttention(
            d=d,
            h=h,
            drop=drop,
            K=cfg["K"],
            L=cfg["L"],
            M=cfg["M"],
            qkv_bias=qkv_bias,
            device=device,
        )

        # ---- Tiny 2-layer MLP gate ----
        # Input: token representation x_t in R^d
        # Output: 2 logits per token -> sigmoid -> [g_hyper, g_race]
        gate_hidden = cfg.get("gate_hidden_dim", max(32, d // 2))

        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        # Optional: normalize gates so they sum to ~1
        # Start with False for NSA-style independent gates.
        self.normalize_gates = cfg.get("gate_normalize", False)

        # Optional debug hook: stores last gates
        self.last_gates = None

    def forward(self, x, mask):
        """
        x   : [B, T, d]
        mask: [B, T]
        returns:
            out : [B, T, d]
        """

        # Branch outputs
        out_hyper = self.hyper(x, mask)   # [B, T, d]
        out_race  = self.race(x, mask)    # [B, T, d]

        # Gate logits from current token representation
        gate_logits = self.gate_mlp(x)    # [B, T, 2]
        gates = torch.sigmoid(gate_logits)  # [B, T, 2]

        # Optional normalization if needed
        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        # Save for debugging / inspection
        self.last_gates = gates.detach()

        # Split the two scalar gates
        g_hyper = gates[..., 0:1]   # [B, T, 1]
        g_race  = gates[..., 1:2]   # [B, T, 1]

        # Weighted sum
        out = g_hyper * out_hyper + g_race * out_race

        # Zero pad positions for cleanliness
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        return out
def hidden_before_layer(model, tokens, mask, layer_idx: int):
    """
    Returns the hidden states BEFORE layer_idx is applied.
    So for layer_idx=0, this is just embeddings+positional embeddings+dropout.
    """
    B, T = tokens.shape
    pos = torch.arange(T, device=tokens.device).unsqueeze(0)

    h = model.tok_emb(tokens) + model.pos_emb(pos)
    h = model.drop(h)

    for i in range(layer_idx):
        h = model.layers[i](h, mask)

    return h


##Agan New Task
@torch.no_grad()
def inspect_random_race_query_bucket(
    model,
    dataset,
    k: int = 64,
    last_q: int = 1000,
    layer_idx: int = 0,
    table_idx: int = 0,
    head_idx: int = 0,
    sample_idx: int | None = None,
    device: str = DEVICE,
):
    """
    Pick one random sample from dataset, inspect one random query token from the
    last_q valid positions, and print:

      - chosen sample index / label
      - query position
      - bucket probabilities for that query (for one chosen table)
      - winning bucket and its probability
      - top-k key positions with highest probability of belonging to that bucket

    Notes
    -----
    - With K=4 and L=4, this inspects ONE table with 16 buckets (2^4 = 16),
      matching your requested intuition experiment.
    - Use test_ds (augment=False) if you want stable readable results.
    """
    model.eval()

    if sample_idx is None:
        sample_idx = random.randrange(len(dataset))

    label, ids = dataset[sample_idx]
    tokens = ids.unsqueeze(0).to(device)                      # [1,T]
    mask = (tokens != dataset.pad_idx).long().to(device)     # [1,T]

    blk = model.layers[layer_idx]
    if not hasattr(blk, "att") or not isinstance(blk.att, RACEAttention):
        raise ValueError(f"Layer {layer_idx} is not a RACE layer.")

    # hidden state entering this block
    h = hidden_before_layer(model, tokens, mask, layer_idx)
    h_in = blk.norm1(h)   # this is what the attention actually sees

    att = blk.att
    B, T, _ = h_in.shape

    # Q/K/V exactly as in RACEAttention.forward
    Q = att.q(h_in).view(B, T, att.h, att.dk)
    K = att.k(h_in).view(B, T, att.h, att.dk)
    V = att.v(h_in).view(B, T, att.h, att.dk)

    m = mask.unsqueeze(-1).unsqueeze(-1).to(Q.dtype)
    Q, K, V = Q * m, K * m, V * m

    probsQ, probsK = race_bucket_probs_from_qk(att, Q, K)   # [M,B,H,T,L,R]
    probsQ = probsQ[0, 0, head_idx]                         # [T,L,R]  (M=1 path)
    probsK = probsK[0, 0, head_idx]                         # [T,L,R]

    valid_T = int(mask[0].sum().item())
    q_start = max(0, valid_T - last_q)
    query_positions = torch.arange(q_start, valid_T, device=device)

    if query_positions.numel() == 0:
        raise ValueError("No valid query positions in the requested window.")

    # choose ONE random query token from the last_q positions
    rand_idx = torch.randint(query_positions.numel(), (1,), device=device)
    q_pos = int(query_positions[rand_idx].item())

    # bucket distribution for this single query token in one chosen table
    q_bucket_probs = probsQ[q_pos, table_idx]      # [R=16]
    win_bucket = int(q_bucket_probs.argmax().item())
    win_prob = float(q_bucket_probs[win_bucket].item())

    # rank all keys by their probability of belonging to the winning bucket
    key_scores = probsK[:valid_T, table_idx, win_bucket]    # [valid_T]
    k_eff = min(k, valid_T)
    top_scores, top_idx = torch.topk(key_scores, k=k_eff, largest=True)

    # also inspect the whole last_q query window on average
    window_mean_bucket_probs = probsQ[query_positions, table_idx].mean(dim=0)  # [R]
    window_win_bucket = int(window_mean_bucket_probs.argmax().item())
    window_win_prob = float(window_mean_bucket_probs[window_win_bucket].item())

    # how much total winning-bucket key mass is captured by top-k?
    mass_fraction = float(top_scores.sum().item() / (key_scores.sum().item() + 1e-12))

    itos = {idx: tok for tok, idx in dataset.stoi.items()}

    print("=" * 90)
    print("RACE bucket inspection")
    print("=" * 90)
    print(f"sample_idx       : {sample_idx}")
    print(f"label            : {label}")
    print(f"valid length     : {valid_T}")
    print(f"layer_idx        : {layer_idx}")
    print(f"head_idx         : {head_idx}")
    print(f"table_idx        : {table_idx}   (16 buckets because K=4)")
    print(f"query window     : [{q_start}, {valid_T})  size={valid_T - q_start}")
    print(f"chosen query pos : {q_pos}")
    print(f"single-query winning bucket : {win_bucket}")
    print(f"single-query bucket prob    : {win_prob:.6f}")
    print(f"window-mean winning bucket  : {window_win_bucket}")
    print(f"window-mean bucket prob     : {window_win_prob:.6f}")
    print(f"top-k key mass fraction in winning bucket: {mass_fraction:.6f}")
    print("-" * 90)

    print("Top-5 bucket probabilities for the chosen query:")
    topb_vals, topb_idx = torch.topk(q_bucket_probs, k=min(5, q_bucket_probs.numel()))
    for rank, (b, p) in enumerate(zip(topb_idx.tolist(), topb_vals.tolist()), start=1):
        print(f"  #{rank:>2} bucket={b:<2d} prob={p:.6f}")

    print("-" * 90)
    print(f"Top-{k_eff} keys for bucket {win_bucket} (position, prob, token_id, token_str):")
    for pos, score in zip(top_idx.tolist(), top_scores.tolist()):
        tok_id = int(tokens[0, pos].item())
        tok_str = itos.get(tok_id, "<unk>")
        print(f"  pos={pos:<6d}  prob={score:.6f}  token_id={tok_id:<6d}  token={tok_str}")

    print("=" * 90)

    return {
        "sample_idx": sample_idx,
        "label": int(label),
        "valid_length": valid_T,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "table_idx": table_idx,
        "query_pos": q_pos,
        "query_window_start": q_start,
        "single_query_bucket_probs": q_bucket_probs.detach().cpu(),
        "single_query_winning_bucket": win_bucket,
        "single_query_winning_prob": win_prob,
        "window_mean_bucket_probs": window_mean_bucket_probs.detach().cpu(),
        "window_mean_winning_bucket": window_win_bucket,
        "window_mean_winning_prob": window_win_prob,
        "topk_positions": top_idx.detach().cpu(),
        "topk_scores": top_scores.detach().cpu(),
        "mass_fraction": mass_fraction,
    }


# ---- FAVOR+ (Performer) ----
def favorplus_features(x, proj, eps=1e-6):
    xw = torch.einsum("bhtd,hmd->bhtm", x, proj)
    xw = xw - xw.max(dim=-1, keepdim=True).values
    exp_part  = torch.exp(xw)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    base      = torch.exp(-0.5 * x_norm_sq)
    return exp_part * base + eps
class HyperLSHExactBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]

        self.att = HyperLSHExactAttention(
            d=d,
            h=h,
            drop=drop,
            num_bits=cfg.get("hyper_num_bits", 5),            # default 32 buckets
            block_size=cfg.get("hyper_block_size", 256),
            min_seq_len=cfg.get("hyper_min_seq_len", 4096),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
class FavorPlusAttention(nn.Module):
    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
        super().__init__()
        assert d % h == 0
        self.h  = h
        self.dk = d // h
        self.m  = m_features

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        if seed is not None:
            torch.manual_seed(seed)
        proj = torch.nn.init.orthogonal_(torch.randn(h, m_features, self.dk))
        self.register_buffer("proj", proj)
        self.eps = 1e-6

    def forward(self, x, mask=None):
        B, T, d = x.shape
        h, dk, m = self.h, self.dk, self.m

        Q = self.q(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        Qs = Q / math.sqrt(dk)
        Ks = K / math.sqrt(dk)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Ks   = Ks * keep
            V    = V  * keep

        phiQ = favorplus_features(Qs, self.proj, eps=self.eps) / math.sqrt(m)
        phiK = favorplus_features(Ks, self.proj, eps=self.eps) / math.sqrt(m)

        if mask is not None:
            keep_m = mask[:, None, :, None].to(phiK.dtype)
            phiK   = phiK * keep_m

        KV   = torch.einsum("bhtm,bhtd->bhmd", phiK, V)
        Ksum = phiK.sum(dim=2)

        num = torch.einsum("bhtm,bhmd->bhtd", phiQ, KV)
        den = torch.einsum("bhtm,bhm->bht",   phiQ, Ksum).unsqueeze(-1) + self.eps
        out_heads = num / den

        merged = out_heads.transpose(1, 2).contiguous().view(B, T, h * dk)
        merged = self.drop(merged)
        return self.o(merged)

# ---- Linear attention (ELU kernel) ----
class LinearAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.eps = eps

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def kernel(self, x):
        return F.elu(x) + 1

    def forward(self, x, mask=None):
        B, T, _ = x.size()
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2)
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            K = K * keep
            V = V * keep

        Q = self.kernel(Q)
        K = self.kernel(K)

        KV = torch.einsum("bhtd,bhte->bhde", K, V)  # [B,H,D,D]
        K_sum = K.sum(dim=2)                       # [B,H,D]

        Z = torch.einsum("bhtd,bhd->bht", Q, K_sum) + self.eps
        context = torch.einsum("bhtd,bhde->bhte", Q, KV)
        out = context / Z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.dropout(out)
        return self.out_proj(out)

# ---- Linformer attention ----
class LinformerAttention(nn.Module):
    def __init__(self, d, dropout, num_heads, qkv_bias, k_proj_dim, max_seq_len):
        super().__init__()
        assert d % num_heads == 0
        self.h  = num_heads
        self.dk = d // num_heads
        self.k_proj_dim = k_proj_dim
        self.max_seq_len = max_seq_len

        self.W_query = nn.Linear(d, d, bias=qkv_bias)
        self.W_key   = nn.Linear(d, d, bias=qkv_bias)
        self.W_value = nn.Linear(d, d, bias=qkv_bias)

        self.E_k = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        self.E_v = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        assert T <= self.max_seq_len
        h, dk, k = self.h, self.dk, self.k_proj_dim

        Q = self.W_query(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.W_key(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            K = K * keep
            V = V * keep

        Ek = self.E_k[:T]  # (T,k)
        Ev = self.E_v[:T]

        K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)  # [B,h,k,dk]
        V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)

        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale
        attn = F.softmax(scores, dim=-1)

        ctx = torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)
        out = ctx.transpose(1, 2).contiguous().view(B, T, h * dk)
        out = self.dropout(out)
        return self.out_proj(out)

# ==================================================
# 6) Transformer blocks (one per baseline)
# ==================================================
class SoftmaxBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att  = MultiHeadAttention(d, h, drop, qkv_bias)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
class ExactFlashBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]

        self.att = ExactFlashAttention(
            d=d,
            h=h,
            drop=drop,
            qkv_bias=qkv_bias,
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att  = AngularAttention(d, h, drop, qkv_bias)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class RACEBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att = RACEAttention(
            d=d, h=h, drop=drop,
            K=cfg["K"], L=cfg["L"], M=cfg["M"],
            qkv_bias=qkv_bias, device=device,
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
#new
class HyperRaceGatedBlock(nn.Module):
    """
    Standard pre-norm transformer block using the hybrid attention above.
    """
    def __init__(self, cfg, device=DEVICE):
        super().__init__()

        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperRaceGatedAttention(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        self.ff = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        # Attention sublayer
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        # FFN sublayer
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h

        return x
class LSHSparseBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]

        self.att = LSHBucketSparseAttention(
            d=d,
            h=h,
            drop=drop,
            num_bits=cfg.get("lsh_num_bits", 5),               # default 32 buckets
            num_hashes=cfg.get("lsh_num_hashes", 2),           # average over 2 hash rounds
            neighbor_buckets=cfg.get("lsh_neighbor_buckets", 1),
            q_chunk_size=cfg.get("lsh_q_chunk_size", 1024),
            qkv_bias=qkv_bias,
            device=device,
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
class LinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        h = cfg["num_heads"]
        self.att  = LinearAttention(
            d_in=d, d_out=d, dropout=drop, num_heads=h, qkv_bias=qkv_bias
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class LinformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        h = cfg["num_heads"]
        k_proj_dim = 128
        self.att  = LinformerAttention(
            d=d,
            dropout=drop,
            num_heads=h,
            qkv_bias=qkv_bias,
            k_proj_dim=k_proj_dim,
            max_seq_len=cfg["max_len"],
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class PerformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        self.att = FavorPlusAttention(
            d=d,
            h=h,
            m_features=cfg["m_features"],
            drop=drop,
            qkv_bias=cfg["qkv_bias"],
            seed=cfg["favor_seed"],
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
# ==================================================
# hyper_race_mexact + hyper_race_lambda_dependent
# Text / mask-aware versions for arxiv_64k.py
# ==================================================

class HyperLSHExactWithLogDenomAttention(nn.Module):
    """
    Text-version of Hyper-LSH exact sparse attention that also returns
    a log denominator proxy:

        log_d_exact_i = log sum_{j in LSH-support(i)} exp(q_i^T k_j / sqrt(d_head))

    Returns:
        out_hyper         : [B,T,d]
        log_d_exact_token : [B,T,1]
    """

    def __init__(
        self,
        d,
        h,
        drop,
        num_bits=5,
        block_size=256,
        min_seq_len=4096,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
        mexact_eps=1e-6,
    ):
        super().__init__()
        assert d % h == 0

        self.h = h
        self.dk = d // h
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks
        self.mexact_eps = mexact_eps
        self.scale = 1.0 / math.sqrt(self.dk)

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.dk, device=device)

    def _full_sdpa_fallback_with_lse(self, Qh, Kh, Vh):
        out_h = _run_exact_sdpa(
            Qh.unsqueeze(0),
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]

        with torch.no_grad():
            logits = torch.einsum("hqd,hkd->hqk", Qh, Kh) * self.scale
            lse_h = torch.logsumexp(logits.float(), dim=-1).to(Qh.dtype)

        return out_h, lse_h

    def _same_block_exact_with_lse(self, Qs, Ks, Vs, valid_T):
        H, _, D = Qs.shape
        bsz = self.block_size

        num_full_blocks = valid_T // bsz
        rem = valid_T % bsz

        out_sorted = torch.zeros_like(Qs)
        lse_sorted = torch.empty(H, valid_T, device=Qs.device, dtype=Qs.dtype)

        if num_full_blocks > 0:
            T_full = num_full_blocks * bsz

            Q_full = Qs[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)
            K_full = Ks[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)
            V_full = Vs[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)

            Q_flat = Q_full.reshape(H * num_full_blocks, 1, bsz, D)
            K_flat = K_full.reshape(H * num_full_blocks, 1, bsz, D)
            V_flat = V_full.reshape(H * num_full_blocks, 1, bsz, D)

            O_flat = _run_exact_sdpa(Q_flat, K_flat, V_flat)
            O_full = O_flat.reshape(H, num_full_blocks, bsz, D).reshape(H, T_full, D)
            out_sorted[:, :T_full, :] = O_full

            # Chunk LSE computation to avoid memory spikes.
            with torch.no_grad():
                lse_chunks = []
                block_chunk = 32
                for bs in range(0, num_full_blocks, block_chunk):
                    be = min(bs + block_chunk, num_full_blocks)
                    logits = torch.einsum(
                        "hnqd,hnkd->hnqk",
                        Q_full[:, bs:be, :, :],
                        K_full[:, bs:be, :, :],
                    ) * self.scale
                    lse_blk = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                    lse_chunks.append(lse_blk)

                lse_full = torch.cat(lse_chunks, dim=1).reshape(H, T_full)
                lse_sorted[:, :T_full] = lse_full

        if rem > 0:
            q_last = Qs[:, num_full_blocks * bsz:, :]
            k_last = Ks[:, num_full_blocks * bsz:, :]
            v_last = Vs[:, num_full_blocks * bsz:, :]

            o_last = _run_exact_sdpa(
                q_last.unsqueeze(0),
                k_last.unsqueeze(0),
                v_last.unsqueeze(0),
            )[0]
            out_sorted[:, num_full_blocks * bsz:, :] = o_last

            with torch.no_grad():
                logits = torch.einsum("hqd,hkd->hqk", q_last, k_last) * self.scale
                lse_last = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                lse_sorted[:, num_full_blocks * bsz:] = lse_last

        return out_sorted, lse_sorted

    def _neighbor_block_exact_with_lse(self, Qs, Ks, Vs, valid_T):
        H, _, D = Qs.shape
        bsz = self.block_size
        num_blocks = math.ceil(valid_T / bsz)

        out_sorted = torch.zeros_like(Qs)
        lse_sorted = torch.empty(H, valid_T, device=Qs.device, dtype=Qs.dtype)

        for bi in range(num_blocks):
            q0 = bi * bsz
            q1 = min((bi + 1) * bsz, valid_T)

            left = max(0, bi - self.neighbor_blocks)
            right = min(num_blocks - 1, bi + self.neighbor_blocks)

            k0 = left * bsz
            k1 = min((right + 1) * bsz, valid_T)

            q_blk = Qs[:, q0:q1, :]
            k_blk = Ks[:, k0:k1, :]
            v_blk = Vs[:, k0:k1, :]

            o_blk = _run_exact_sdpa(
                q_blk.unsqueeze(0),
                k_blk.unsqueeze(0),
                v_blk.unsqueeze(0),
            )[0]
            out_sorted[:, q0:q1, :] = o_blk

            with torch.no_grad():
                logits = torch.einsum("hqd,hkd->hqk", q_blk, k_blk) * self.scale
                lse_blk = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                lse_sorted[:, q0:q1] = lse_blk

        return out_sorted, lse_sorted

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        H, D = self.h, self.dk

        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q = Q * keep
            K = K * keep
            V = V * keep

        out_heads = torch.zeros_like(Q)
        lse_heads = torch.full(
            (B, H, T),
            float("-inf"),
            device=x.device,
            dtype=Q.dtype,
        )

        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue

            Qh = Q[b, :, :valid_T, :]
            Kh = K[b, :, :valid_T, :]
            Vh = V[b, :, :valid_T, :]

            if valid_T < self.min_seq_len:
                out_b, lse_b = self._full_sdpa_fallback_with_lse(Qh, Kh, Vh)
                out_heads[b, :, :valid_T, :] = out_b
                lse_heads[b, :, :valid_T] = lse_b
                continue

            q_bucket_ids = self.lsh.hash(Qh)
            k_bucket_ids = self.lsh.hash(Kh)

            q_sort_idx = torch.argsort(q_bucket_ids, dim=1, stable=True)
            k_sort_idx = torch.argsort(k_bucket_ids, dim=1, stable=True)
            q_sort_inv = torch.argsort(q_sort_idx, dim=1, stable=True)

            Qs = _gather_tokens_3d(Qh, q_sort_idx)
            Ks = _gather_tokens_3d(Kh, k_sort_idx)
            Vs = _gather_tokens_3d(Vh, k_sort_idx)

            if self.neighbor_blocks == 0:
                O_sorted, LSE_sorted = self._same_block_exact_with_lse(Qs, Ks, Vs, valid_T)
            else:
                O_sorted, LSE_sorted = self._neighbor_block_exact_with_lse(Qs, Ks, Vs, valid_T)

            O_unsorted = O_sorted.gather(
                1,
                q_sort_inv.unsqueeze(-1).expand(-1, -1, D),
            )
            LSE_unsorted = LSE_sorted.gather(1, q_sort_inv)

            out_heads[b, :, :valid_T, :] = O_unsorted
            lse_heads[b, :, :valid_T] = LSE_unsorted

        with torch.no_grad():
            log_d_exact_token = (
                torch.logsumexp(lse_heads.float(), dim=1)
                - math.log(H)
            ).to(Q.dtype).unsqueeze(-1)

        out = out_heads.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.drop(out)
        out = self.o(out)

        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        return out, log_d_exact_token


class RACEAttentionWithDenom(nn.Module):
    """
    Text-version RACE branch that returns:
        out_race      : [B,T,d]
        d_race_token  : [B,T,1]

    It uses the same RACE parameters/ACE mechanics as your existing RACEAttention,
    but masks PAD keys out of the denominator statistics.
    """

    def __init__(self, d, h, drop, K, L, M, qkv_bias=False, device="cpu"):
        super().__init__()
        assert d % h == 0

        self.h = h
        self.dk = d // h
        self.M = M

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        self.ace = BatchedACE(self.dk, K, L, M, device=device)

    def _ace_with_denom(self, Khf, Vhf, Qhf, mask=None, eps=1e-6):
        ace = self.ace

        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.dk

        S = ace.L * ace.R
        scale = ace.logit_temp.exp().clamp(1e-2, 20.0)

        if ace.share_planes:
            N = M * B * H

            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            projK = Kh2 @ ace.planes_T
            projQ = Qh2 @ ace.planes_T

        else:
            BH = B * H

            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            projK = torch.einsum("mbtd,mds->mbts", Kh2, ace.planes_T)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, ace.planes_T)

            projK = projK.contiguous().view(M * BH, T, ace.L * ace.K)
            projQ = projQ.contiguous().view(M * BH, T, ace.L * ace.K)
            V2    = V2.contiguous().view(M * BH, T, dk)
            N     = M * BH

        projK = projK.view(N, T, ace.L, ace.K)
        projQ = projQ.view(N, T, ace.L, ace.K)

        logitsK = (projK.tanh().div(scale) @ ace.protos_T)
        logitsQ = (projQ.tanh().div(scale) @ ace.protos_T)

        probsK = F.softmax(logitsK, dim=-1)
        probsQ = F.softmax(logitsQ, dim=-1)

        probsK_S = probsK.contiguous().view(N, T, S)
        probsQ_S = probsQ.contiguous().view(N, T, S)

        if mask is not None:
            mask_bh = (
                mask[:, None, :]
                .expand(B, H, T)
                .contiguous()
                .view(B * H, T)
            )
            mask_N = (
                mask_bh.unsqueeze(0)
                .expand(M, -1, -1)
                .contiguous()
                .view(N, T)
                .to(probsK_S.dtype)
            )

            probsK_S = probsK_S * mask_N.unsqueeze(-1)
            probsQ_S = probsQ_S * mask_N.unsqueeze(-1)
            V2 = V2 * mask_N.unsqueeze(-1)

        total_num = probsK_S.transpose(1, 2).bmm(V2)  # [N,S,dk]
        total_den = probsK_S.sum(dim=1)               # [N,S]

        E = total_num / (total_den.unsqueeze(-1) + eps)
        out2 = probsQ_S.bmm(E)                        # [N,T,dk]

        d2 = torch.einsum("nts,ns->nt", probsQ_S, total_den).clamp_min(eps)

        # arxiv_64k.py BatchedACE layout is [M,B,H,T,dk]
        out = out2.view(M, B, H, T, dk).contiguous()
        den = d2.view(M, B, H, T).contiguous()

        return out, den

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        H, dk, M = self.h, self.dk, self.M

        Q = self.q(x).view(B, T, H, dk)
        K = self.k(x).view(B, T, H, dk)
        V = self.v(x).view(B, T, H, dk)

        if mask is not None:
            keep = mask[:, :, None, None].to(Q.dtype)
            Q = Q * keep
            K = K * keep
            V = V * keep

        def pack(z):
            return z.unsqueeze(0).expand(M, -1, -1, -1, -1)

        out_m, den_m = self._ace_with_denom(pack(K), pack(V), pack(Q), mask=mask)

        out_heads = out_m.mean(dim=0)  # [B,H,T,dk]
        out = out_heads.permute(0, 2, 1, 3).contiguous().view(B, T, H * dk)

        d_race_heads = den_m.mean(dim=0)  # [B,H,T]
        d_race_token = d_race_heads.mean(dim=1).unsqueeze(-1).clamp_min(1e-6)

        out = self.drop(self.o(out))

        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        return out, d_race_token


class HyperRaceMExactAttention(nn.Module):
    """
    hyper_race_mexact for text:

        m_exact = d_exact / (d_exact + lambda * d_race + eps)

    lambda is a positive scalar:
        lambda = exp(log_lambda)

    If mexact_lambda_learnable=False, lambda is fixed to 1.
    """

    def __init__(self, cfg, device=DEVICE):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 64)

        self.mexact_eps = cfg.get("mexact_eps", 1e-6)

        self.mexact_lambda_learnable = bool(cfg.get("mexact_lambda_learnable", False))
        lambda_init = float(cfg.get("mexact_lambda_init", 1.0))
        if lambda_init <= 0:
            raise ValueError("mexact_lambda_init must be > 0.")

        if self.mexact_lambda_learnable:
            self.log_mexact_lambda = nn.Parameter(
                torch.tensor(math.log(lambda_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_mexact_lambda",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=False,
            )

        self.hyper = HyperLSHExactWithLogDenomAttention(
            d=d,
            h=h,
            drop=drop,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 256),
            min_seq_len=cfg.get("hyper_min_seq_len", 4096),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
            mexact_eps=self.mexact_eps,
        )

        self.race = RACEAttentionWithDenom(
            d=d,
            h=h,
            drop=drop,
            K=cfg["K"],
            L=cfg["L"],
            M=cfg["M"],
            qkv_bias=qkv_bias,
            device=device,
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.normalize_gates = cfg.get("gate_normalize", False)

        self.last_gates = None
        self.last_m_exact = None
        self.last_d_exact_mean = None
        self.last_d_race_mean = None
        self.last_mexact_lambda = None

    def forward(self, x, mask=None):
        out_hyper, log_d_exact = self.hyper(x, mask)
        out_race, d_race = self.race(x, mask)

        log_d_exact_det = log_d_exact.detach().float()
        log_d_race_det = torch.log(d_race.detach().float().clamp_min(self.mexact_eps))

        log_lambda = self.log_mexact_lambda.float()
        log_eps = torch.full_like(log_d_exact_det, math.log(self.mexact_eps))

        log_den = torch.logsumexp(
            torch.stack(
                [
                    log_d_exact_det,
                    log_lambda + log_d_race_det,
                    log_eps,
                ],
                dim=0,
            ),
            dim=0,
        )

        m_exact = torch.exp(log_d_exact_det - log_den).to(out_hyper.dtype)

        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)

        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        g_hyper = gates[..., 0:1]
        g_race = gates[..., 1:2]

        out = g_hyper * m_exact * out_hyper + g_race * out_race

        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        self.last_gates = gates.detach()
        self.last_m_exact = m_exact.detach()
        self.last_mexact_lambda = self.log_mexact_lambda.detach().exp()

        if mask is not None and bool(mask.any().item()):
            valid = mask.bool().unsqueeze(-1)
            de = torch.exp(log_d_exact.detach().clamp(max=20.0))
            dr = d_race.detach()
            self.last_d_exact_mean = de[valid].mean()
            self.last_d_race_mean = dr[valid].mean()
        else:
            self.last_d_exact_mean = torch.exp(log_d_exact.detach().clamp(max=20.0)).mean()
            self.last_d_race_mean = d_race.detach().mean()

        return out


class HyperRaceLambdaDependentAttention(nn.Module):
    """
    hyper_race_lambda_dependent for text:

        lambda_i = c + sigmoid(w^T q_i + b)

        m_exact_i =
            d_exact_i / (d_exact_i + lambda_i * d_race_i + eps)

    c can be fixed or learnable.
    """

    def __init__(self, cfg, device=DEVICE):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 64)

        self.mexact_eps = cfg.get("mexact_eps", 1e-6)
        self.lambda_min = float(cfg.get("mexact_dependent_lambda_min", 1e-6))

        self.lambda_offset_learnable = bool(
            cfg.get("mexact_dependent_lambda_offset_learnable", False)
        )
        self.lambda_offset_positive = bool(
            cfg.get("mexact_dependent_lambda_offset_positive", True)
        )

        offset_init = float(cfg.get("mexact_dependent_lambda_offset", 0.3))

        if self.lambda_offset_learnable:
            self.lambda_offset_raw = nn.Parameter(
                torch.tensor(offset_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "lambda_offset_raw",
                torch.tensor(offset_init, dtype=torch.float32),
                persistent=False,
            )

        self.lambda_use_bias = bool(cfg.get("mexact_dependent_lambda_use_bias", True))
        self.lambda_detach_q = bool(cfg.get("mexact_dependent_lambda_detach_q", True))

        lambda_w_init_std = float(cfg.get("mexact_dependent_lambda_w_init_std", 1e-3))
        if lambda_w_init_std < 0:
            raise ValueError("mexact_dependent_lambda_w_init_std must be >= 0.")

        self.lambda_w = nn.Parameter(torch.empty(d, dtype=torch.float32))
        nn.init.normal_(self.lambda_w, mean=0.0, std=lambda_w_init_std)

        if self.lambda_use_bias:
            effective_offset_init = max(offset_init, 0.0) if self.lambda_offset_positive else offset_init
            init_target = float(
                cfg.get("mexact_dependent_lambda_init_target", effective_offset_init + 0.5)
            )
            init_prob = init_target - effective_offset_init
            init_prob = min(max(init_prob, 1e-4), 1.0 - 1e-4)

            init_bias = math.log(init_prob / (1.0 - init_prob))
            self.lambda_bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        else:
            self.register_buffer(
                "lambda_bias",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=False,
            )

        self.hyper = HyperLSHExactWithLogDenomAttention(
            d=d,
            h=h,
            drop=drop,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 256),
            min_seq_len=cfg.get("hyper_min_seq_len", 4096),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
            mexact_eps=self.mexact_eps,
        )

        self.race = RACEAttentionWithDenom(
            d=d,
            h=h,
            drop=drop,
            K=cfg["K"],
            L=cfg["L"],
            M=cfg["M"],
            qkv_bias=qkv_bias,
            device=device,
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.normalize_gates = cfg.get("gate_normalize", False)

        self.last_gates = None
        self.last_m_exact = None
        self.last_d_exact_mean = None
        self.last_d_race_mean = None
        self.last_mexact_lambda = None
        self.last_mexact_lambda_logits = None
        self.last_mexact_lambda_offset = None
        self.last_mexact_lambda_sigmoid = None

    def _current_lambda_offset(self, dtype, device):
        c_raw = self.lambda_offset_raw.to(device=device, dtype=dtype)

        if self.lambda_offset_positive:
            c_forward = c_raw.clamp_min(0.0)
            c = c_raw + (c_forward - c_raw).detach()
        else:
            c = c_raw

        return c

    def _compute_query_dependent_lambda(self, x):
        if self.lambda_detach_q:
            with torch.no_grad():
                q_for_lambda = self.hyper.q(x)
        else:
            q_for_lambda = self.hyper.q(x)

        q_for_lambda = q_for_lambda.float()

        lambda_logits = torch.matmul(q_for_lambda, self.lambda_w.float())

        if self.lambda_use_bias:
            lambda_logits = lambda_logits + self.lambda_bias.float()

        lambda_sigmoid = torch.sigmoid(lambda_logits)

        lambda_offset = self._current_lambda_offset(
            dtype=lambda_sigmoid.dtype,
            device=lambda_sigmoid.device,
        )

        lambda_q = lambda_offset + lambda_sigmoid
        lambda_q = lambda_q.clamp_min(self.lambda_min).unsqueeze(-1)

        return lambda_q, lambda_logits, lambda_offset, lambda_sigmoid

    def forward(self, x, mask=None):
        out_hyper, log_d_exact = self.hyper(x, mask)
        out_race, d_race = self.race(x, mask)

        lambda_q, lambda_logits, lambda_offset, lambda_sigmoid = (
            self._compute_query_dependent_lambda(x)
        )

        log_d_exact_det = log_d_exact.detach().float()
        log_d_race_det = torch.log(d_race.detach().float().clamp_min(self.mexact_eps))
        log_lambda_q = torch.log(lambda_q.float().clamp_min(self.mexact_eps))
        log_eps = torch.full_like(log_d_exact_det, math.log(self.mexact_eps))

        log_den = torch.logsumexp(
            torch.stack(
                [
                    log_d_exact_det,
                    log_lambda_q + log_d_race_det,
                    log_eps,
                ],
                dim=0,
            ),
            dim=0,
        )

        m_exact = torch.exp(log_d_exact_det - log_den).to(out_hyper.dtype)

        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)

        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        g_hyper = gates[..., 0:1]
        g_race = gates[..., 1:2]

        out = g_hyper * m_exact * out_hyper + g_race * out_race

        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        self.last_gates = gates.detach()
        self.last_m_exact = m_exact.detach()
        self.last_mexact_lambda = lambda_q.detach()
        self.last_mexact_lambda_logits = lambda_logits.detach()
        self.last_mexact_lambda_offset = lambda_offset.detach()
        self.last_mexact_lambda_sigmoid = lambda_sigmoid.detach()

        if mask is not None and bool(mask.any().item()):
            valid = mask.bool().unsqueeze(-1)
            de = torch.exp(log_d_exact.detach().clamp(max=20.0))
            dr = d_race.detach()
            self.last_d_exact_mean = de[valid].mean()
            self.last_d_race_mean = dr[valid].mean()
        else:
            self.last_d_exact_mean = torch.exp(log_d_exact.detach().clamp(max=20.0)).mean()
            self.last_d_race_mean = d_race.detach().mean()

        return out


class HyperRaceMExactBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperRaceMExactAttention(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class HyperRaceLambdaDependentBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperRaceLambdaDependentAttention(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


# Optional alias, so both names work.
HyperRaceDependentLambdaBlock = HyperRaceLambdaDependentBlock
# ==================================================
# 7) Text Transformer classifier (ViT-style structure)
# ==================================================
class TextTransformerClassifier(nn.Module):
    def __init__(self, cfg, attn_type: str):
        super().__init__()
        self.cfg = cfg
        vocab_size = cfg["vocab_size"]
        max_len    = cfg["max_len"]
        d          = cfg["embed_dim"]

        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.drop    = nn.Dropout(cfg["drop_rate"])

       
        if attn_type == "softmax":
            Block = SoftmaxBlock
        elif attn_type == "exact_flash":
            Block = ExactFlashBlock    
        elif attn_type == "race":
            Block = lambda c: RACEBlock(c, device=DEVICE)
        elif attn_type == "lsh_sparse":
            Block = lambda c: LSHSparseBlock(c, device=DEVICE)
        elif attn_type == "hyper_race_mexact":
            Block = lambda c: HyperRaceMExactBlock(c, device=DEVICE)
        elif attn_type in {"hyper_race_lambda_dependent", "hyper_race_dependent_lambda"}:
            Block = lambda c: HyperRaceLambdaDependentBlock(c, device=DEVICE)    
        elif attn_type == "hyper_race":
            Block = lambda c: HyperRaceGatedBlock(c, device=DEVICE)    
        elif attn_type == "hyper_lsh":
            Block = lambda c: HyperLSHExactBlock(c, device=DEVICE)        
        elif attn_type == "angular":
            Block = AngularBlock
        elif attn_type == "linear":
            Block = LinearBlock
        elif attn_type == "linformer":
            Block = LinformerBlock
        elif attn_type == "performer":
            Block = PerformerBlock
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        self.layers = nn.ModuleList(
            [Block(cfg) for _ in range(cfg["num_layers"])]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["num_classes"])

    def forward(self, x, mask):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.layers:
            h = blk(h, mask)
        h = self.norm(h)
        # CLS-style: use position 0
        logits = self.head(h[:, 0])
        return logits

# ==================================================
# 8) Scheduler & training loop (like vision file)
# ==================================================
class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (1.0 - progress)
            lrs.append(lr)
        return lrs
def _new_running_stat():
    return {
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "count": 0,
    }


def _update_running_stat(stats, name, tensor):
    if tensor is None:
        return

    t = tensor.detach().float().reshape(-1).cpu()
    if t.numel() == 0:
        return

    if name not in stats:
        stats[name] = _new_running_stat()

    st = stats[name]
    st["sum"] += t.sum().item()
    st["sumsq"] += (t ** 2).sum().item()
    st["min"] = min(st["min"], t.min().item())
    st["max"] = max(st["max"], t.max().item())
    st["count"] += t.numel()


def _finalize_running_stats(stats):
    logs = {}

    for name, st in stats.items():
        count = max(st["count"], 1)
        mean = st["sum"] / count
        var = max(st["sumsq"] / count - mean ** 2, 0.0)
        std = math.sqrt(var)

        logs[f"{name}_mean"] = mean
        logs[f"{name}_std"] = std
        logs[f"{name}_min"] = st["min"]
        logs[f"{name}_max"] = st["max"]

    return logs


def _collect_hybrid_attention_stats(model, attn_type, stats):
    if attn_type not in {
        "hyper_race",
        "hyper_race_mexact",
        "hyper_race_lambda_dependent",
        "hyper_race_dependent_lambda",
    }:
        return

    for layer_idx, layer in enumerate(model.layers):
        if not hasattr(layer, "att"):
            continue

        att = layer.att

        if hasattr(att, "last_gates") and att.last_gates is not None:
            gates = att.last_gates
            _update_running_stat(stats, f"gates/layer{layer_idx}_hyper", gates[..., 0])
            _update_running_stat(stats, f"gates/layer{layer_idx}_race", gates[..., 1])

        if hasattr(att, "last_m_exact") and att.last_m_exact is not None:
            _update_running_stat(stats, f"m_exact/layer{layer_idx}", att.last_m_exact)

        if hasattr(att, "last_d_exact_mean") and att.last_d_exact_mean is not None:
            _update_running_stat(stats, f"den/layer{layer_idx}_exact", att.last_d_exact_mean)

        if hasattr(att, "last_d_race_mean") and att.last_d_race_mean is not None:
            _update_running_stat(stats, f"den/layer{layer_idx}_race", att.last_d_race_mean)

        if hasattr(att, "last_mexact_lambda") and att.last_mexact_lambda is not None:
            _update_running_stat(stats, f"lambda/layer{layer_idx}", att.last_mexact_lambda)

        if hasattr(att, "last_mexact_lambda_logits") and att.last_mexact_lambda_logits is not None:
            _update_running_stat(stats, f"lambda_logits/layer{layer_idx}", att.last_mexact_lambda_logits)

        if hasattr(att, "last_mexact_lambda_offset") and att.last_mexact_lambda_offset is not None:
            _update_running_stat(stats, f"lambda_offset/layer{layer_idx}", att.last_mexact_lambda_offset)

        if hasattr(att, "last_mexact_lambda_sigmoid") and att.last_mexact_lambda_sigmoid is not None:
            _update_running_stat(stats, f"lambda_sigmoid/layer{layer_idx}", att.last_mexact_lambda_sigmoid)
def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    cfg,
    attn_type,
    grad_accum_steps: int = 1,
):
    steps_per_epoch   = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates     = num_epochs * updates_per_epoch
    warmup_updates    = max(1, int(0.1 * total_updates))

    scheduler = LinearWarmupLR(
        optimizer,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
    )

    out_path = f"arxiv_{attn_type}_644K.txt"

    def _log(fp, msg):
        print(msg)
        fp.write(msg + "\n")
        fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Attn: {attn_type}, Epochs: {num_epochs}")
        _log(f, "-" * 80)
        global_update = 0

        for epoch in range(1, num_epochs + 1):
            # ---- TRAIN ----
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t0 = time.time()

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            running_correct = 0
            running_total = 0
            accum_count = 0

            train_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch} [train]",
                leave=False,
            )

            for tokens, masks, labels in train_iter:
                tokens  = tokens.to(device)
                masks   = masks.to(device)
                labels  = labels.to(device)

                logits = model(tokens, masks)
                loss   = F.cross_entropy(logits, labels)

                (loss / grad_accum_steps).backward()
                accum_count += 1

                preds = logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
                running_total   += labels.size(0)
                running_loss    += loss.item()

                # Update tqdm with running stats
                train_iter.set_postfix({
                    "loss": running_loss / max(1, len(train_iter)),
                    "acc":  running_correct / max(1, running_total),
                })

                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            if accum_count > 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if "cuda" in str(device):
                torch.cuda.synchronize()
            train_time = time.time() - t0

            tr_l = running_loss / len(train_loader)
            tr_a = running_correct / max(1, running_total)

            # ---- VAL ----
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t1 = time.time()

            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total   = 0
            hybrid_stats = {}
            val_iter = tqdm(
                val_loader,
                desc=f"Epoch {epoch} [val]",
                leave=False,
            )

            with torch.no_grad():
                for tokens, masks, labels in val_iter:
                    tokens = tokens.to(device)
                    masks  = masks.to(device)
                    labels = labels.to(device)

                    logits = model(tokens, masks)
                    loss   = F.cross_entropy(logits, labels)
                    val_loss_total += loss.item()

                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)
                    _collect_hybrid_attention_stats(model, attn_type, hybrid_stats)
                    val_iter.set_postfix({
                        "loss": val_loss_total / max(1, len(val_iter)),
                        "acc":  val_correct / max(1, val_total),
                    })

            if "cuda" in str(device):
                torch.cuda.synchronize()
            val_time = time.time() - t1

            va_l = val_loss_total / len(val_loader)
            va_a = val_correct / max(1, val_total)
            curr_lr = scheduler.get_last_lr()[0]
            extra_logs = _finalize_running_stats(hybrid_stats)            
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_l,
                "train/acc": tr_a,
                "val/loss": va_l,
                "val/acc": va_a,
                "lr": curr_lr,
                "time/train_sec": train_time,
                "time/val_sec": val_time,
                **extra_logs,
            }, step=epoch)
            _log(
                f,
                (f"Ep{epoch:3d} | "
                 f"train_loss {tr_l:.4f}, acc {tr_a:.4f} ({train_time:.1f}s) | "
                 f"val_loss {va_l:.4f}, acc {va_a:.4f} ({val_time:.1f}s) | "
                 f"lr {curr_lr:.3e} | updates {global_update}/{total_updates}")
            )

        _log(f, "-" * 80)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")


# ==================================================
# 9) Run all baselines (like vision)
# ==================================================
def run_experiment(attn_types, cfg):
    for attn_type in attn_types:
        run = wandb.init(
            project="RACE",
            name=f"{cfg.get('task_mode', 'arxiv_classification')}_{attn_type}_{cfg['max_len']}",
            config={
                "attn_type": attn_type,
                "max_len": cfg["max_len"],
                "embed_dim": cfg["embed_dim"],
                "num_heads": cfg["num_heads"],
                "num_layers": cfg["num_layers"],
                "mlp_dim": cfg["mlp_dim"],
                "batch_size": cfg["batch_size"],
                "epochs": cfg["epochs"],
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "grad_accum_steps": cfg["grad_accum_steps"],
                "K": cfg["K"],
                "L": cfg["L"],
                "M": cfg["M"],
                "mexact_eps": cfg["mexact_eps"],
                "mexact_lambda_learnable": cfg["mexact_lambda_learnable"],
                "mexact_lambda_init": cfg["mexact_lambda_init"],

                "mexact_dependent_lambda_offset": cfg["mexact_dependent_lambda_offset"],
                "mexact_dependent_lambda_offset_learnable": cfg["mexact_dependent_lambda_offset_learnable"],
                "mexact_dependent_lambda_offset_positive": cfg["mexact_dependent_lambda_offset_positive"],
                "mexact_dependent_lambda_init_target": cfg["mexact_dependent_lambda_init_target"],
                "mexact_dependent_lambda_use_bias": cfg["mexact_dependent_lambda_use_bias"],
                "mexact_dependent_lambda_detach_q": cfg["mexact_dependent_lambda_detach_q"],
                "mexact_dependent_lambda_min": cfg["mexact_dependent_lambda_min"],
                "mexact_dependent_lambda_w_init_std": cfg["mexact_dependent_lambda_w_init_std"],
                "pack_target_len": PACK_TARGET_LEN,
                "min_doc_len": MIN_DOC_LEN,
                "pack_min_frac": PACK_MIN_FRAC,
                "desired_train_total": DESIRED_TRAIN_TOTAL,
                "desired_test_total": DESIRED_TEST_TOTAL,
                "task_mode": cfg.get("task_mode", "arxiv_classification"),
                "retrieval_pair_len": cfg.get("retrieval_pair_len", None),
                "retrieval_train_pairs": cfg.get("retrieval_train_pairs", None),
                "retrieval_test_pairs": cfg.get("retrieval_test_pairs", None),
                "retrieval_batch_size": cfg.get("retrieval_batch_size", None),
            }
        )

        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("time/*", step_metric="epoch")
        wandb.define_metric("val/acc", summary="max")
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("gates/*", step_metric="epoch")
        wandb.define_metric("m_exact/*", step_metric="epoch")
        wandb.define_metric("den/*", step_metric="epoch")
        wandb.define_metric("lambda/*", step_metric="epoch")
        wandb.define_metric("lambda_logits/*", step_metric="epoch")
        wandb.define_metric("lambda_offset/*", step_metric="epoch")
        wandb.define_metric("lambda_sigmoid/*", step_metric="epoch")

        print(f"\n=== Training {attn_type.upper()} on Arxiv {cfg['max_len']} ===")
        model = TextTransformerClassifier(cfg, attn_type).to(DEVICE)
        opt   = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        train_model_simple(
            model=model,
            train_loader=train_dl,
            val_loader=test_dl,
            optimizer=opt,
            device=DEVICE,
            num_epochs=cfg["epochs"],
            cfg=cfg,
            attn_type=attn_type,
            grad_accum_steps=cfg["grad_accum_steps"],
        )
        wandb.finish()

if __name__ == "__main__":
    run_experiment(
        [           
            #"hyper_race_mexact",
            "hyper_race_lambda_dependent",
            #"hyper_race",
            #"race",
            "hyper_lsh",
            #"exact_flash",
            
            ],  #race "hyper_race","hyper_lsh" ,"linear", "linformer", "performer","exact_flash", "angular" , "softmax" , "lsh_sparse"
        TEXT_CONFIG,
    )
