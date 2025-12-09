# Repository Guidelines

## Project Structure & Modules
- Core Python package lives in `drrm/` (policies, diffusion models, dataset utilities). Entry script is `main.py` driven by Hydra configs.
- Training and eval configs are under `configs/` (per-task YAML) and `configs/accelerate_config.yaml` (hardware/launch settings).
- Shell helpers in `scripts/` (e.g., `scripts/train_demo.sh`, `scripts/eval/robotwin_exp/`) wrap common launch patterns.
- Simulation assets and Robotwin adapters are under `simulation/` (see `simulation/robotwin/script/` for eval drivers; `simulation/robotwin2.0/` for the latest simulator).
- Data and outputs are expected in `datasets/`, `checkpoints/`, and `outputs/`; third-party deps live in `third_party/`.

## Build, Test, and Development Commands
- Install dev environment: `pip install -e .` (after activating the `drrm` conda env noted in README).
- Train (example VODP):  
  `accelerate launch --config_file configs/accelerate_config.yaml main.py --config-path configs/vodp_train --config-name vodp_23d_1f.yaml train_dataset.path=datasets/lerobot_D435_200 train_dataset.task=block_hammer_beat train_dataset.demo=100`
- Evaluate in Robotwin (VODP):  
  `python simulation/robotwin/script/eval_policy_vodp.py --checkpoint-dir YOUR/CHECKPOINT/DIR --save-dir YOUR/SAVE/DIR --task-name TASK_NAME --num-process 8 --seed 0`
- Light checks: prefer `python -m unittest discover drrm "*test*.py"` when adding unit tests; no test runner is pre-wired, so keep tests self-contained.

## Coding Style & Naming
- Python 3.10+, PEP 8 with 4-space indentation; keep modules snake_case and classes CapWords.
- Favor type hints and clear tensor shapes in docstrings; mirror existing Hydra config naming (snake_case YAML keys).
- Keep configs declarative (no side effects); avoid hard-coding dataset paths—use Hydra overrides.

## Testing Guidelines
- Add unit tests near the code they cover or under `drrm/tests/`; use `unittest` from stdlib to avoid extra deps.
- For policy/eval changes, include a small dry-run log (e.g., single-step launch with `--num-process 1`) and note required assets.
- When touching Hydra configs, validate with `python -m drrm.configs.validate` if you add a checker, or at minimum run one `accelerate` launch with `--dry-run` where applicable.

## Commit & Pull Request Guidelines
- Commit messages follow a short prefix pattern seen in history (`feat: ...`, `fix: ...`, `docs: ...`, `chore:` or concise `Update:`); keep imperative tone.
- PRs should describe the change, list tested commands, and call out any dataset/model checkpoints required. Attach logs or screenshots for sim runs when relevant.
- Note configuration impacts (modified YAMLs) and backward-compatibility considerations; link related issues or experiment trackers if available.
