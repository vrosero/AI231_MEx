from datasets import load_dataset

ds = load_dataset(
    "airimonda/ai231-me2-voice-commands",
    cache_dir="/data/ai231",
)
train, test, holdout = ds["train"], ds["test"], ds["holdout"]
numerals = ds["numerals"]                                   # number-only clips, separate from the splits
extra = load_dataset("airimonda/ai231-me2-voice-commands", 
                     "supplemental_synth",
                     cache_dir="/data/ai231"
                     )["train"]   # unused synthetic clips; see voice_split
negs = load_dataset("airimonda/ai231-me2-voice-commands", 
                    "synthetic_negatives", 
                    cache_dir="/data/ai231")   # generated out-of-scope clips: negs["train"], negs["test"]
