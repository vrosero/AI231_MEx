# 02 · Dataset — HF `airimonda/ai231-me2-voice-commands` + supplementals

!(figures/fig_dataset.png)

## 1. Overview

The corpus is the Hugging Face dataset
[`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
plus its two supplemental file sets, merged at load time. Every clip is a short
spoken command (or a deliberate out-of-scope negative); audio is stored at 16 kHz
and resampled to the 12 kHz front-end rate in the loader. The merged corpus holds
**36,604 clips / 21.0 h** of audio.

Every row carries `audio, file, transcript, command, variation, slot_value,
out_of_scope, bucket, speaker_id, source, is_synthetic, accent_group, numerals,
duration_s, transcript_source, whisper_check, note, variation_match` — enough to
rebuild any mixture without re-downloading.

## 2. Splits

| Split | Clips | Hours | Base repo | + supp_synth (voice_split) | + synthetic_negatives | + supp_fil |
|---|---:|---:|---:|---:|---:|---:|
| Train | **29,836** | 17.3 | 10,733 | +3,983 | +1,000 | +14,120 |
| Test | **6,305** | 3.4 | 4,443 | +1,612 | +250 | — |
| Holdout | **463** | 0.3 | 202 | +261 | — | — |
| **Total** | **36,604** | 21.0 | 15,378 | +5,856 | +1,250 | +14,120 |

* The repo's `train/test/holdout` parquets are **speaker-disjoint** (no speaker in
  two splits), so test numbers measure generalisation to new voices.
* `supplemental_synth` (5,856 unused synthetic clips) is routed by its own
  `voice_split` column, preserving the disjointness.
* `supplemental_fil` (14,120 clips, `fil50_persona`) is **train-only** by design —
  a persona voice bank that must not leak into evaluation.
* `synthetic_negatives` (1,000 train + 250 test) are generated out-of-scope clips
  (noise, babble, reversed, truncated, near-silence) that teach the model to
  reject non-commands.
* The repo's `numerals` split (66,390 MLEnd number-only clips) is **excluded** —
  the command set has no slot that needs spoken digits.

Training carves a speaker-disjoint validation set out of train (5 % = 987 clips,
33 speakers), leaving **28,849 effective training clips**.

## 3. Sources (merged train split)

| Source | Clips | Character |
|---|---:|---|
| `fil50_persona` (supplemental_fil) | 14,120 | persona voice bank, train-only |
| `group_synthetic` | 12,288 | the class collective's synthetic set — the homogeneous core |
| `SLURP` | 1,010 | public smart-home SLU corpus subset (CC BY 4.0) |
| `synthetic_negative` | 1,000 | generated out-of-scope clips |
| `real_voice` | 620 | the group's own recordings (do not share outside the class) |
| `SNIPS` | 331 | public Snips command subset |
| `FluentSpeechCommands` | 256 | public smart-home command set (non-commercial academic) |
| `xela_SET_TEMPERATURE_REAL` | 60 | real temperature commands (Xela's S1–S5) |
| `CommonVoice_en` | 36 | Mozilla Common Voice 19 (CC0) |
| `TimersAndSuch` | 31 | public timer/reminder corpus |
| `SpeechCommands_v2` | 11 | Google Speech Commands v2 (CC BY 4.0) |
| `xela_Multi-Sensor` | 4 | Multi-Sensor Voice Command (CC BY 4.0, DOI 10.48804/IEKKVZ) |
| `group_synthetic_oos` | 69 | synthetic out-of-scope clips |

Test split composition mirrors it: `group_synthetic` 5,202, `SLURP` 403,
`real_voice` 179, `FluentSpeechCommands` 144, `synthetic_negative` 250,
`SNIPS` 61, and small public-source tails. Holdout (463 clips) is the tiny
never-touched set reserved for the final number.

## 4. Labels

* **37 fine labels** on the merged train split = the dispatchable commands, e.g.
  `ALARM_6_00AM/8_00AM/9_00PM`, `BRIGHTNESS_20/60/100`, `COLOR_RED/BLUE/GREEN/YELLOW`,
  `CREATE_REMINDER_{DRINK_WATER,EXERCISE,STUDY}`, `LIGHT_ON/OFF`, `TIMER_10s/30s/1m`,
  `TEMPERATURE_18/22/26`, `PLAY_MUSIC`, `PAUSE`, `STOP`, `NEXT`, `VOLUME_UP/DOWN`,
  `TIME`, `WEATHER`, `CALL`, `MESSAGE`, `LIST_REMINDERS`, `OUT_OF_SCOPE`, plus the
  slot-less generic variants (`ALARM`, `BRIGHTNESS`, `CREATE_REMINDER`,
  `TEMPERATURE`, `TIMER`) that appear only in training.
* **20 intents** = the coarse action groups the fine labels roll up to
  (`fine2intent` map ships in the model meta). Test/holdout see 32 of the 37 fine
  labels — the 5 slot-less generics are train-only.

## 5. Class balance

Per-intent train counts span **COLOR 3,014 → PAUSE 743** (4.1× skew):

| Band | Intents (train clips) |
|---|---|
| ≥ 2,000 | COLOR 3,014 · TEMPERATURE 2,835 · ALARM 2,717 · TIMER 2,609 · CREATE_REMINDER 2,287 · BRIGHTNESS 2,138 |
| 1,000–2,000 | NEXT 1,792 · VOLUME_UP 1,092 · VOLUME_DOWN 1,075 · PLAY_MUSIC 1,045 · STOP 1,153 · OUT_OF_SCOPE 1,270 |
| < 1,000 | LIGHT_ON 979 · LIGHT_OFF 988 · TIME 981 · WEATHER 856 · LIST_REMINDERS 851 · MESSAGE 787 · PAUSE 743 · CALL 624 |

The supplemental merge is what *levels* this distribution — the base repo's
imbalance was far worse. The report still leads with **macro-F1 and per-intent
recall** (§04) rather than raw accuracy.

## 6. Feature extraction (fixed front-end)

| Field | Value |
|---|---|
| Sample rate | 12 kHz mono (HF clips are 16 kHz, resampled in the loader) |
| Spectrogram | log-mel, 40 bands, n_fft 400, hop 200 |
| Frames | cropped/padded to 96 (~1.6 s window) |
| Normalisation | z-score per clip |
| Padding value | −11.5 |
| Mel range | 50–6,000 Hz |

Features are decoded once into `.npy` shards (`model/cache/`, one pass over the
audio) so training reads memory-mapped arrays instead of parquet. The identical
front-end is re-implemented in `rpi/runtime/features.py` (numpy only); parity with
the training features is exact (max diff 0.0).

## 7. Licenses

Each source keeps its own license (SLURP, Speech Commands v2: CC BY 4.0; Common
Voice 19: CC0; Fluent Speech Commands: non-commercial academic; SNIPS: its own
terms; Timers and Such: other-open; MLEnd numerals: its own terms; Multi-Sensor
Voice Command: CC BY 4.0; group synthetic + group recordings: the class's own —
group recordings may not be shared outside the class without each speaker's
consent). Research and educational use.