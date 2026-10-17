from datasets import load_dataset

REPO = "airimonda/ai231-me2-voice-commands"
CACHE_DIR = "/data/ai231"


def load(name=None):
    """Load a config from the shared cache; raise loudly if it fails."""
    try:
        if name is None:
            return load_dataset(REPO, cache_dir=CACHE_DIR)
        return load_dataset(REPO, name, cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"FAILED to load config {name or 'default'}: {e}")
        raise


def describe_audio(audio):
    """Return (num_samples, sample_rate) for either decoder-style or dict-style audio."""
    if hasattr(audio, "get_all_samples"):          # torchcodec AudioDecoder
        s = audio.get_all_samples()
        return s.data.shape, s.sample_rate
    return audio["array"].shape, audio["sampling_rate"]   # older dict format


# --- main splits ---
ds = load()
train, test, holdout = ds["train"], ds["test"], ds["holdout"]
numerals = ds["numerals"]          # number-only clips, separate from the splits

# --- additional configs ---
extra = load("supplemental_synth")["train"]   # unused synthetic clips; see voice_split
negs = load("synthetic_negatives")            # generated out-of-scope clips
negs_train, negs_test = negs["train"], negs["test"]

print("SUCCESS: all datasets loaded\n")

splits = {
    "train": train, "test": test, "holdout": holdout, "numerals": numerals,
    "supplemental_synth": extra,
    "synthetic_negatives/train": negs_train,
    "synthetic_negatives/test": negs_test,
}

print("Rows and backing files:")
for name, split in splits.items():
    files = split.cache_files[0]["filename"] if split.cache_files else "(in memory)"
    print(f"  {name:26s} rows: {len(split):>7d} | {files}")

print("\nSchema check vs train:")
for name, split in splits.items():
    same = split.column_names == train.column_names
    print(f"  {name:26s} {'same columns' if same else 'DIFFERENT: ' + str(split.column_names)}")

print("\nFirst example (train, metadata only):")
print(train.remove_columns("audio")[0])

print("\nAudio decode check (first clip of each split):")
for name, split in splits.items():
    shape, sr = describe_audio(split[0]["audio"])
    print(f"  {name:26s} shape: {tuple(shape)} | sample rate: {sr}")