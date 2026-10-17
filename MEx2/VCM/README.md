# VCM — Voice Command Module

Voice-controlled device assistant: a CRNN command classifier (37 fine labels)
plus a "Hello Jarvis" wake-word CNN, running fully offline on a Raspberry Pi.
Full write-up: [`report/`](report/) (`VCM_Report.pdf`, `VCM_Slides.pptx`,
`vcm_deck.pptx`).

## Layout

| Folder | What it is |
|---|---|
| `dataset/` | Wake-word data (`Jarvis/`: positive/negative/noise) + command data via HF |
| `model/` | Training code (`vcm_model.py`), checkpoint (`ckpt/crnn.pt`), wake trainer (`wakeword/cnn/`), outputs |
| `rpi/` | The deployable runtime (executable-only: `runtime/`, `model/`, `audio/`, `music/`, `setup.sh`, `run.sh`) |
| `rpi_inf/` | Model-inference measurement: `perf.py`, `check_contract.py`, `gen_responses.py`, `perf_results/` |
| `report/` | Report sources + built deliverables |

## Environments — one per job, never shared

| Venv | Job | Created by |
|---|---|---|
| `.venv_train` | CRNN prepare/train/eval, ONNX, report figures/PDF/decks | `setup_env.sh` (Mac) |
| `.venv_ww` | Wake-CNN trainer — torch 2.8, the stack that produced the deployed `jarvis_cnn.onnx` | `setup_env.sh` (Mac) |
| `.venv_inf` | `rpi_inf/`: perf bench/wake, contract check, response regen | `setup_env.sh` (Mac) |
| `rpi/.venv` | Pi runtime — created **on the Pi** by `rpi/setup.sh`; never on the Mac | `rpi/setup.sh` (Pi) |

All Mac venvs are pinned (`requirements-*.txt`) and git-ignored.

## Quickstart

```bash
bash setup_env.sh              # once — creates the three Mac venvs
bash run_all.sh                # full pipeline: prepare -> train -> deploy -> bench -> docs (~3 h)
bash run_all.sh --skip-train   # reuse ckpt/crnn.pt, redo eval/ONNX/bench/docs
python rpi_inf/check_contract.py   # with .venv_inf active: feature + torch<->ONNX parity
```

Deploy (from the parent of `VCM/`):

```bash
ssh megumi_rpi@megumi-rpi.local 'mkdir -p ~/vcm'
scp -r VCM/rpi megumi_rpi@megumi-rpi.local:~/vcm
scp -r VCM/rpi_inf megumi_rpi@megumi-rpi.local:~/vcm
# on the Pi: cd ~/vcm && bash setup.sh && bash run.sh
```