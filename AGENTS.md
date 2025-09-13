# AGENTS.md

> Operating guide for human + AI agents on the **CNN–SNN Hybrid (PyTorch + snnTorch)** project.

## 0) Project Snapshot

- **Goal:** Hybrid CNN→SNN image classifier on MNIST / Fashion-MNIST. CNN extracts spatial features; LIF spiking head (snnTorch) provides temporal dynamics.
- **Success:** ≥97% test accuracy (CNN baseline); hybrid within ~1–3% at T≈10–25 timesteps; spike raster plots; reproducible runs.
- **Env:** WSL (Ubuntu) + VS Code, Python **3.11**, project-local **.venv**, PyTorch, snnTorch.
- **Scaffold:** Spec-Kit (`specify init …`) with `/specify`, `/plan`, `/tasks` prompts to drive work.

Repo layout (expected):
```
specs/001-cnn-snn-hybrid/{spec.md, plan.md, tasks.md}
src/{model.py, train.py, raster.py}
data/   results/   .vscode/   .gitignore   README.md   requirements.txt   AGENTS.md
```

---

## 1) Roles & Responsibilities

- **/specify (Spec Author)**  
  Defines *what & why*. Align with course slides; scope, success criteria, risks.
- **/plan (Planner / Architect)**  
  Chooses tech stack, architecture, hyperparams, training/eval, file layout.
- **/tasks (Project Manager)**  
  Breaks into actionable steps; marks done/in-progress; logs results (acc, T).
- **Implementer (Coder)**  
  Creates/edits files exactly as planned; keeps code runnable; writes docstrings.
- **Maintainer (Ops)**  
  Manages env, dependencies, formatting, tests, CI, version pinning, backups.
- **Reviewer (QA)**  
  Sanity checks outputs, plots, metrics; validates reproducibility and README.

> One human can wear multiple hats; AI agents should be told which hat they’re wearing in each prompt.

---

## 2) Ground Rules (for all agents)

- **Work in the project venv** (`.venv`) and do **not** install project deps globally.
- **Never overwrite** data or results unless told; always write to `results/` with unique names.
- **Create only the files requested**; follow the exact paths and names.
- **Keep code self-contained:** minimal assumptions, clear imports, `if __name__ == "__main__":` guards.
- **Respect file boundaries:**  
  - `src/model.py` → architectures (CNNOnly, CNNSNN).  
  - `src/train.py` → dataloaders, training loops, CLI args, saving artifacts.  
  - `src/raster.py` → visualization helpers (e.g., spike raster).
- **Document run steps** in `README.md` whenever behavior changes.
- **Log results** (accuracy, timesteps, hyperparams) to `results/metrics.json`.

---

## 3) One-Line Commands for Agents (use in Copilot/Claude/Cursor Chat)

### Spec-Kit prompts
- **Create/refresh spec**  
  ```
  /specify Build a hybrid CNN→SNN image classifier for MNIST (optional Fashion-MNIST) using PyTorch+snnTorch. Deliver working code + training/eval, spike raster plots, and a README. Out of scope: neuromorphic hardware (optional Jetson later). Success: ≥97% CNN baseline; hybrid within ~1–3% at T=10–25.
  ```
- **Create/refresh plan**  
  ```
  /plan Env: WSL Ubuntu + VS Code; Python 3.11 venv. Stack: PyTorch, snnTorch, torchvision, matplotlib. Model: Conv(1→12,k=5)-MaxPool-LIF(β≈0.9) → Conv(12→64,k=5)-MaxPool-LIF → Flatten → Linear(64*4*4→10)-LIF. Train: ce_rate_loss, Adam lr=1e-3, T=20. Baseline: CNN-only. Outputs: accuracy vs baseline, spike rasters, results/*.pt, metrics.json. Layout: specs/001-…/, src/, data/, results/.
  ```
- **Create/refresh tasks**  
  ```
  /tasks Setup venv & deps; Baseline CNN (train/eval, save results/cnn_baseline.pt); Hybrid (add LIF, time loop T=10–25, ce_rate_loss, tune β/lr/T, save results/hybrid.pt + metrics.json + spike rasters); Regularization (dropout; optional aug for FMNIST); Visualization (curves, rasters); Docs (README updates); Optional: Jetson inference script + latency notes.
  ```

### Implementation prompts
- **Generate model file**  
  ```
  /implement Create src/model.py with CNNOnly (2×Conv→ReLU→Pool→FC→10) and CNNSNN using snnTorch LIF (β≈0.9); provide forward_step(x, mem1, mem2, mem3) and init_state(x); no training code.
  ```
- **Generate training script**  
  ```
  /implement Create src/train.py with argparse (--model cnn|hybrid, --epochs, --batch, --timesteps, --lr), MNIST loaders, CrossEntropy for CNN, ce_rate_loss for hybrid (time loop over T), evaluation, and saving to results/cnn_baseline.pt, results/hybrid.pt, and results/metrics.json.
  ```
- **Generate spike raster helper**  
  ```
  /implement Create src/raster.py that loads results/hybrid.pt, runs CNNSNN for T=20 on one test sample, stacks output spikes over time, and plots a spike raster with matplotlib.
  ```
- **Update README**  
  ```
  /implement Update README.md Quickstart (create venv, install requirements, train CNN & Hybrid, plot raster), and list artifacts produced under results/.
  ```

---

## 4) Terminal Playbook (human or agent shell runner)

- **Create & activate venv; install deps; pin versions**
  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install torch torchvision snntorch matplotlib tqdm numpy && pip freeze > requirements.txt
  ```
- **Train baseline**
  ```bash
  python src/train.py --model cnn --epochs 3
  ```
- **Train hybrid**
  ```bash
  python src/train.py --model hybrid --epochs 3 --timesteps 20
  ```
- **Plot spike raster**
  ```bash
  python src/raster.py
  ```
- **Freeze exact versions**
  ```bash
  pip freeze > requirements.txt
  ```

---

## 5) Code Style & Quality

- **Python:** PEP8; docstrings on public functions/classes; type hints where feasible.
- **Logging:** Print epoch loss and final test accuracy; keep outputs concise.
- **Reproducibility:** Set seeds where used; save models under `results/`.
- **Formatting/Linting (optional):**
  - `black src` and/or `ruff check --fix src`
- **Commits:** Imperative mood, scoped (e.g., `train: add ce_rate_loss loop`, `model: lif head`).

---

## 6) Safety & Guardrails for AI Agents

- Do **not** install global packages; project deps go in `.venv`.
- Do **not** rewrite unrelated files; only touch requested paths.
- Do **not** hard-code local absolute paths; use relative paths.
- Do **not** delete `results/` contents automatically.
- If unsure, **ask for confirmation** with a concrete diff or file list.

---

## 7) Review Checklist (before calling a task “Done”)

- [ ] Code runs in a clean venv with `pip install -r requirements.txt`.
- [ ] `python src/train.py --model cnn` trains and saves `results/cnn_baseline.pt`.
- [ ] `python src/train.py --model hybrid --timesteps 20` trains and saves `results/hybrid.pt` + `results/metrics.json`.
- [ ] Spike raster renders from `src/raster.py`.
- [ ] `README.md` reflects current CLI and artifacts.
- [ ] `spec.md`/`plan.md`/`tasks.md` updated to match reality (acc, T, β).

---

## 8) Troubleshooting Quickies

- **`specify: command not found`** → `pipx ensurepath && exec $SHELL` (ensure `~/.local/bin` in PATH).
- **`ModuleNotFoundError: snntorch`** → activate venv (`source .venv/bin/activate`), then `pip install snntorch`.
- **VS Code using wrong Python** → `Ctrl+Shift+P` → *Python: Select Interpreter* → pick the project’s `.venv`.
- **CUDA missing** (optional) → falls back to CPU; accuracy should still meet targets (training slower).

---

## 9) Future Work (optional)

- Jetson inference script (`src/infer_jetson.py`), measure latency/FPS.
- ANN→SNN conversion (SpikingJelly) experiments.
- CI smoke test (GitHub Actions) to run a 1-epoch CNN and Hybrid on CPU.
