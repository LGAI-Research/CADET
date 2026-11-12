# CADET: Construction of PCB Routing Instances with Executable Demonstrations

[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-FFD21E)](https://huggingface.co/LG-AI-Research/CADET)
[![PCBWorld](https://img.shields.io/badge/PCBWorld-v1.0.1-blue)](https://github.com/LGAI-Research/PCBWorld/tree/v1.0.1)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green)](LICENSE)

Code of CADET, from LG AI Research.

CADET builds PCB routing instances together with demonstrations that are executable in the routing engine, trains a
policy on them by behavior cloning (BC), and fine-tunes it with PPO.

- **Trained policies**: the six checkpoints behind the result tables are on Hugging Face,
  [LG-AI-Research/CADET](https://huggingface.co/LG-AI-Research/CADET) (download: [Setup](#setup)).
- **Environment**: [PCBWorld](https://github.com/LGAI-Research/PCBWorld) **v1.0.1** (cited in the paper). Its
  environment, router engine, PPO trainer and rule-based baselines are used as they are.

This repository holds

- the board and demonstration generator (RouteDemos) with its TryRoute search,
- the fresh-engine replay verification,
- BC and PPO fine-tuning, with the from-scratch PPO references of the ablation,
- the evaluation on real boards (D3-B test, D3-A) and the table scripts, including the rule-based, LLM and
  human-reference rows,
- the D3-B ref / test split and the reference statistics.

Contents: [Results](#results) · [Layout](#layout) · [Setup](#setup) · [Pipeline](#pipeline) · [Licensing](#licensing) · [Contact](#contact)

## Results

Real test boards, from the paper's main table. CADET is averaged over four training seeds.

**D3-B test (141 boards)**

| Method | Type | CP (%) | ΔΦ | Rout. (%) | Time (s/board) |
|---|---|---:|---:|---:|---:|
| FreeRouting @1 | Rule-based | 65.6 | 119.33 | **96.8** | 45.6 |
| GPT-6 Astra @1 | LLM | 56.0 | 104.20 | 86.0 | 491.7 |
| GPT-6 Luna @1 | LLM | 5.0 | 37.72 | 47.9 | 335.7 |
| CADET @1 | BC+RL | 67.2 | 107.22 | 91.7 | **15.5** |
| CADET @3 | BC+RL | **79.8** | **121.30** | 95.9 | 47.6 |

**D3-A (99 boards)**

| Method | Type | CP (%) | ΔΦ | Rout. (%) | Time (s/board) |
|---|---|---:|---:|---:|---:|
| FreeRouting @1 | Rule-based | 82.1 | 46.05 | 97.7 | 8.6 |
| GPT-6 Astra @1 | LLM | 83.8 | 45.03 | 93.8 | 138.9 |
| GPT-6 Luna @1 | LLM | 46.5 | 31.30 | 78.5 | 179.1 |
| CADET @1 | BC+RL | 89.5 | 45.13 | 96.6 | **4.2** |
| CADET @3 | BC+RL | **96.3** | **48.87** | **99.2** | 12.3 |

- **CP** (clean pass): the fraction of boards with all connections completed and zero error-severity DRC violations.
- **ΔΦ**: the potential gain over the unrouted board.
- **Rout.** (routability): the fraction of required connections completed.
- **@k**: the best of k rollouts.

Bold marks the best value per column of each table. The paper's table also has the human reference, KRT and the PPO
baseline.

## Layout

```
cadet/
  paths.py, base_args.py   locations of this repository / the PCBWorld checkout; loader of the settings files
  generation/   run_generation.sh, generate.py (generator), density calibration, selection, reports
  search/       executor.py, executor2.py, grid_oracle.py (TryRoute search); replay_demo.py, replay_shard.py (replay check)
  stats/        fit_board_params.py (reference statistics), make_d3_half_split.py, fidelity_extract.py, fidelity_tables.py
  bc/           behavior cloning: run_bc.sh, train.py, data/index/replay/objective/evaluate, demos_to_bc_run.py
  rl/           launch.sh (PPO cells), val_watcher.py, make_split_json.py
  eval/         rollout_queue.py, prepare_cell.py, score_cell.sh, table.py, bootstrap.py
  llm/          LLM baseline: apply_patch.sh, run_llm.sh
  configs/      base_args/ (environment and model settings), fit_params/, splits/
  tests/
checkpoints/    policy weights and their training arguments, downloaded from Hugging Face (not in this repository)
setup.sh, env.sh   PCBWorld installation and the environment of every command below
```

## Setup

Requirements: Linux x86_64, Python 3.12, one CUDA GPU for training and evaluation. Generation and replay run on CPU.

**Install**

```bash
bash setup.sh                 # clones PCBWorld v1.0.1, creates the conda env `cadet`, builds the engine
conda activate cadet
source ./env.sh
python -m pytest cadet/tests -q
```

`setup.sh` accepts `PCBWORLD_ENV=<name>` for another env name, and `PCBWORLD_ROOT=<checkout>` for an existing PCBWorld v1.0.1 installation (activate that installation's env instead).

**Checkpoints**

```bash
hf download LG-AI-Research/CADET --local-dir checkpoints    # the six policy checkpoints
```

**Data**

- Real boards (D3): follow PCBWorld's README ("Open-source boards (D3)", `tools/quickstart/prepare_pcbench.sh`) so
  that `$PCBWORLD_DATA_ROOT/pcbench/exacad_sorted/` holds the processed boards.
- D2: PCBWorld's synthetic set `synthetic/synth_2L_v2`. Its README ("Train an RL policy") generates it; the D2 cell
  uses 10,000 training boards.

## Pipeline

All commands run from the repository root after `source ./env.sh`. Outputs go to `var/`.

### 1. Split and reference statistics

The split and the statistics used in the paper are in the repository:

| File | Content |
|---|---|
| `cadet/configs/splits/d3_medium_half.json` | the split of the D3-B boards into D3-B ref (142) and D3-B test (141) |
| `cadet/configs/splits/boards_d3b_fit.txt`, `boards_d3b_test.txt` | the two board lists |
| `cadet/configs/splits/boards_d3bf.txt` | an extra D3-B list that the trainer command names for a fifth validation slot; the watcher skips it and no result uses it |
| `cadet/configs/fit_params/d3b_fit_half.json` | the statistics of D3-B ref that the generator samples from (boards of at most 1 cm² are filtered out) |

To rebuild them:

```bash
python cadet/stats/make_d3_half_split.py --out /tmp/split.json --lists-dir /tmp      # rebuild the split
python cadet/stats/fit_board_params.py --split test --out /tmp/d3b_test_half.json    # extract statistics
```

### 2. RouteDemos generation and replay verification

```bash
GATE=1 OUT=var/datasets/routedemos        bash cadet/generation/run_generation.sh    # with TryRoute: boards + demonstrations
GATE=0 OUT=var/datasets/routedemos_nogate bash cadet/generation/run_generation.sh    # ablation: without TryRoute
```

The script

1. calibrates the pad density,
2. generates the boards,
3. keeps the first N accepted boards in seed order,
4. replays every kept training demonstration in a fresh engine (`cadet/search/replay_shard.py`),
5. writes `RESULTS.md`.

The defaults are the paper's settings: 100,000 training boards, 128 validation and test boards, and a per-board time
limit of 1500 s. The paper's run took a median of 106.5 s per accepted board.

A small run, which skips the calibration and uses the paper's density scale:

```bash
OUT=var/datasets/routedemos_small N_TRAIN=200 N_VAL=16 SCALE=1.6505 BOARD_TIME_CAP=300 \
    bash cadet/generation/run_generation.sh
```

`BOARD_TIME_CAP` is the per-board time limit in seconds.

### 3. Fidelity statistics

```bash
python cadet/stats/fidelity_extract.py --routedemos var/datasets/routedemos/train \
    --d2 $PCBWORLD_DATA_ROOT/synthetic/synth_2L_v2/train --out var/results/fidelity
python cadet/stats/fidelity_tables.py var/results/fidelity
```

### 4. Behavior cloning

```bash
DATA=var/datasets/routedemos OUT=var/checkpoints/bc bash cadet/bc/run_bc.sh           # 4 GPUs x batch 128, 350,000 steps
```

Architecture and observation settings come from `cadet/configs/base_args/bc.json`. The checkpoint at step 280,000
initializes fine-tuning (`checkpoints/cadet_bc.pt`).

### 5. PPO fine-tuning and from-scratch references

```bash
bash cadet/rl/launch.sh ours          # CADET: BC checkpoint -> PPO, 200 iterations, KL penalty 0.05
GPU=1 bash cadet/rl/launch.sh wours   # its validation watcher, on a second GPU
```

| cell | paper row | training boards | iterations | checkpoint |
|---|---|---|---|---|
| `ours` | CADET | RouteDemos | 200 | `checkpoints/cadet.pt` (160) |
| — | CADET-BC | RouteDemos | BC 280k steps | `checkpoints/cadet_bc.pt` |
| `solvable` | PPO (RouteDemos) | RouteDemos | 500 | `checkpoints/ppo_routedemos.pt` (460) |
| `unsolvable` | PPO (w/o TryRoute) | RouteDemos w/o TryRoute | 500 | `checkpoints/ppo_routedemos_no_tryroute.pt` (460) |
| `d2a_big` | PPO (D2) | D2 | 500 | `checkpoints/ppo_d2.pt` (170) |
| `real_fit` | PPO (real) | D3-B ref | 500 | `checkpoints/ppo_real.pt` (460) |

The number in parentheses is the reported iteration. `cadet_bc.pt` holds the BC weights, stored with the fine-tuning
run's evaluation settings.

- **Resources**: each cell uses one training GPU and 64 environments; the watcher evaluates saved checkpoints every
  10 iterations.
- **Checkpoint selection**: the reported checkpoint of a cell is the one with the highest `fp_mean_of_means` on
  `val_d3b_fit` (D3-B ref). D3-B test is never used for selection.
- **Dry run**: `DRY_RUN=1` prints the full trainer command of a cell.

### 6. Evaluation on real boards

```bash
# rollouts: 12 samples per board, one environment (serial timing), bf16
python cadet/eval/rollout_queue.py --ckpt checkpoints/cadet.pt --boards-list cadet/configs/splits/boards_d3b_test.txt \
    --prefix val_d3b_test --n-rollouts 12 --n-envs 1 --bf16 --save-boards --out var/runs/cadet_d3b_test
# cell + post-hoc DRC scoring with PCBWorld's evaluation pipeline
python cadet/eval/prepare_cell.py rl var/runs/cadet_d3b_test var/cells/cadet_d3b_test
bash cadet/eval/score_cell.sh var/cells/cadet_d3b_test
# table: @1 = four single samples, @3 = four best-of-three samples
python cadet/eval/table.py --n-boards 141 --init var/cells/cadet_d3b_test \
    --row "CADET @1=var/cells/cadet_d3b_test:@1" --row "CADET @3=var/cells/cadet_d3b_test:@3" \
    --time "CADET @3=episode_wall:var/runs/cadet_d3b_test/episode_wall.jsonl"
```

Variants of the commands above:

- **D3-A**: `cadet/configs/splits/boards_d3a.txt` (99 boards, `--n-boards 99`).
- **Ablation checkpoints**: `--n-rollouts 4 --n-envs 64` and the `@1` row.
- **FreeRouting and KRT**: run with PCBWorld's rule-based runner (its `methods/baselines/rule_based/README.md`), four
  seeds for FreeRouting. The runner writes one `manifest.json` per seed;
  `prepare_cell.py rule_based <manifest> [<manifest> ...] <boards list> <cell dir>` turns them into one cell.
- **Bootstrap intervals**: `bootstrap.py <CADET cell> <FreeRouting cell>` gives the paired bootstrap intervals.

### 7. Human reference and LLM rows

**Human reference.** The original routed boards are scored exactly like a method's output:

```bash
python cadet/eval/prepare_cell.py reference cadet/configs/splits/boards_d3b_test.txt var/cells/reference_d3b_test
bash cadet/eval/score_cell.sh var/cells/reference_d3b_test
python cadet/eval/table.py --n-boards 141 --init var/cells/cadet_d3b_test --row "Human reference=var/cells/reference_d3b_test:0"
```

**LLM rows.** An LLM agent is run through an API: one rollout per board, v7 prompt, history 4, 512 steps, early stop
after 10 steps without geometry progress.

```bash
bash cadet/llm/apply_patch.sh
OPENAI_API_KEY=... API_MODEL=<model id> LIST=cadet/configs/splits/boards_d3b_test.txt OUT=var/runs/llm_d3b_test \
    bash cadet/llm/run_llm.sh                       # paid API calls; resumable; BOARDS="0101 0103" runs a subset
python cadet/eval/prepare_cell.py llm var/runs/llm_d3b_test cadet/configs/splits/boards_d3b_test.txt var/cells/llm_d3b_test
bash cadet/eval/score_cell.sh var/cells/llm_d3b_test
python cadet/eval/table.py --n-boards 141 --init var/cells/cadet_d3b_test \
    --row "LLM @1=var/cells/llm_d3b_test:0" --time "LLM @1=cell:var/cells/llm_d3b_test"
```

## Licensing

The code in this repository is licensed under [BSD-3-Clause](LICENSE). Third-party notices: [Notice.md](Notice.md).

The checkpoints are licensed separately from this code, under the EXAONE AI Model License Agreement 1.2 - NC:
research and educational use only, with no commercial use of the checkpoints or of their output (see the license in the
Hugging Face repository).

## Contact

Open an issue, or write to yj.park@lgresearch.ai or 7590sok@gmail.com.
