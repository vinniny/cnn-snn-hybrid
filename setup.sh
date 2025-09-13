#!/usr/bin/env bash
set -euo pipefail

# Default config (override via flags or env)
PROJECT_NAME="${PROJECT_NAME:-cnn-snn-hybrid}"
PYBIN="${PYBIN:-/usr/bin/python3.11}"
CREATE_SPEC="${CREATE_SPEC:-1}"           # 1 = init spec-kit; 0 = skip
INSTALL_SPEC_KIT="${INSTALL_SPEC_KIT:-1}" # 1 = ensure pipx + spec-kit
VENV_DIR="${VENV_DIR:-.venv}"
PKGS_DEFAULT=(torch torchvision snntorch matplotlib tqdm numpy)
GIT_INIT="${GIT_INIT:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --project-name <name>     Project folder name (default: ${PROJECT_NAME})
  --python <path>           Python 3.11+ interpreter (default: ${PYBIN})
  --venv <dir>              Virtualenv directory (default: ${VENV_DIR})
  --no-spec-kit             Skip installing/initializing Spec-Kit
  --no-git                  Skip git init/first commit
  --help                    Show this help

Examples:
  $(basename "$0") --project-name cnn-snn-hybrid --python /usr/bin/python3.11
EOF
}

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name) PROJECT_NAME="$2"; shift 2 ;;
    --python)       PYBIN="$2";        shift 2 ;;
    --venv)         VENV_DIR="$2";     shift 2 ;;
    --no-spec-kit)  CREATE_SPEC=0; INSTALL_SPEC_KIT=0; shift ;;
    --no-git)       GIT_INIT=0; shift ;;
    --help|-h)      usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# 0) Preconditions
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "ERROR: Python not found at $PYBIN"
  echo "Tip: sudo apt install -y python3.11 python3.11-venv python3.11-distutils"
  exit 1
fi
PYVER=$("$PYBIN" -c 'import sys;print(".".join(map(str, sys.version_info[:3])))')
MAJOR=$("$PYBIN" -c 'import sys;print(sys.version_info[0])')
MINOR=$("$PYBIN" -c 'import sys;print(sys.version_info[1])')
if (( MAJOR < 3 || (MAJOR==3 && MINOR<11) )); then
  echo "ERROR: Python >= 3.11 required (found $PYVER)"
  exit 1
fi

# 1) Create project folder
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# 2) Ensure pipx + Spec-Kit (global CLI) if requested
if (( INSTALL_SPEC_KIT == 1 )); then
  if ! command -v pipx >/dev/null 2>&1; then
    echo "Installing pipx..."
    if command -v sudo >/dev/null 2>&1; then
      sudo apt update && sudo apt install -y pipx || true
    else
      echo "WARN: sudo not available; please install pipx manually if needed."
    fi
    pipx ensurepath || true
  fi
  if ! command -v specify >/dev/null 2>&1; then
    echo "Installing Spec-Kit via pipx..."
    pipx install --python "$PYBIN" 'git+https://github.com/github/spec-kit.git' || true
  else
    echo "Spec-Kit already installed: $(command -v specify)"
  fi
fi

# 3) Create venv (project-local)
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  "$PYBIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# 4) Install Python packages (inside venv)
echo "Installing base packages: ${PKGS_DEFAULT[*]}"
pip install "${PKGS_DEFAULT[@]}"

# 5) Basic repo layout
mkdir -p src data results specs .vscode
touch src/__init__.py

# 6) .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
*.egg
*.pyo

# Virtual envs
.venv/
env/
venv/

# Data & results
data/
results/

# VSCode
.vscode/
EOF

# 7) requirements.txt
pip freeze | sed '/^-e /d' > requirements.txt

# 8) README
if [[ ! -f README.md ]]; then
cat > README.md <<'EOF'
# CNN–SNN Hybrid (PyTorch + snnTorch)

Spec-driven project that trains a hybrid CNN→SNN classifier on MNIST/Fashion-MNIST.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train baseline CNN
python src/train.py --model cnn --epochs 5

# train hybrid CNN→SNN
python src/train.py --model hybrid --epochs 5 --timesteps 20
```

Artifacts go to `results/`.
EOF
fi

# 9) VS Code settings (WSL-friendly)
cat > .vscode/settings.json <<EOF
{
  "python.defaultInterpreterPath": "${PWD}/${VENV_DIR}/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.analysis.autoImportCompletions": true,
  "editor.formatOnSave": true
}
EOF

# 10) Minimal model + train skeletons
if [[ ! -f src/model.py ]]; then
cat > src/model.py <<'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn


class CNNOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 12, 5)
        self.c2 = nn.Conv2d(12, 64, 5)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNSNN(nn.Module):
    def __init__(self, beta: float = 0.9):
        super().__init__()
        self.c1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta)
        self.c2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        self.lif3 = snn.Leaky(beta=beta)

    def forward_step(self, x, mem1, mem2, mem3):
        # conv + pool
        cur1 = F.max_pool2d(self.c1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = F.max_pool2d(self.c2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = self.fc(spk2.view(spk2.size(0), -1))
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3, (mem1, mem2, mem3)

    @torch.no_grad()
    def init_state(self, x):
        # Initialize membrane states with correct shapes derived from a dry pass
        cur1 = F.max_pool2d(self.c1(x), 2)
        mem1 = torch.zeros_like(cur1)
        z1 = torch.zeros_like(cur1)
        cur2 = F.max_pool2d(self.c2(z1), 2)
        mem2 = torch.zeros_like(cur2)
        z2 = torch.zeros_like(cur2)
        cur3 = self.fc(z2.view(z2.size(0), -1))
        mem3 = torch.zeros_like(cur3)
        return (mem1, mem2, mem3)
EOF
fi

if [[ ! -f src/train.py ]]; then
cat > src/train.py <<'EOF'
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import functional as SF
from model import CNNOnly, CNNSNN


def get_loaders(batch: int = 64, root: str = "data"):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=batch, shuffle=False, num_workers=0),
    )


def train_cnn(args, device):
    net = CNNOnly().to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    train_loader, test_loader = get_loaders(args.batch)
    for ep in range(args.epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
        acc = evaluate_cnn(net, test_loader, device)
        print(f"[CNN] epoch {ep+1}/{args.epochs} acc={acc:.4f}")
    save_path = os.path.join("results", "cnn_baseline.pt")
    os.makedirs("results", exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"Saved baseline to {save_path}")


@torch.no_grad()
def evaluate_cnn(net, loader, device):
    net.eval()
    corr = 0
    tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        pred = out.argmax(1)
        corr += (pred == y).sum().item()
        tot += y.numel()
    return corr / max(tot, 1)


def train_hybrid(args, device):
    T = args.timesteps
    net = CNNSNN().to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = SF.ce_rate_loss()
    train_loader, test_loader = get_loaders(args.batch)
    for ep in range(args.epochs):
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            spk_hist = []
            mem = net.init_state(x)
            for _ in range(T):
                spk, mem = net.forward_step(x, *mem)
                spk_hist.append(spk)
            spk_hist = torch.stack(spk_hist)  # [T,B,10]
            loss = loss_fn(spk_hist, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate_hybrid(net, test_loader, device, T)
        print(f"[HYB] epoch {ep+1}/{args.epochs} acc={acc:.4f}")
    os.makedirs("results", exist_ok=True)
    torch.save(net.state_dict(), os.path.join("results", "hybrid.pt"))
    with open(os.path.join("results", "metrics.json"), "w") as f:
        json.dump({"acc": acc, "T": T}, f)
    print("Saved hybrid model + metrics")


@torch.no_grad()
def evaluate_hybrid(net, loader, device, T: int):
    net.eval()
    corr = 0
    tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        rate = torch.zeros(x.size(0), 10, device=device)
        mem = net.init_state(x)
        for _ in range(T):
            spk, mem = net.forward_step(x, *mem)
            rate += spk
        pred = rate.argmax(1)
        corr += (pred == y).sum().item()
        tot += y.numel()
    return corr / max(tot, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "hybrid"], default="cnn")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    if args.model == "cnn":
        train_cnn(args, device)
    else:
        train_hybrid(args, device)


if __name__ == "__main__":
    main()
EOF
fi

# 11) Makefile for common tasks
cat > Makefile <<'EOF'
.PHONY: venv install cnn hybrid freeze clean fmt lint spec
PY?=.venv/bin/python
PIP?=.venv/bin/pip

venv:
	@test -d .venv || /usr/bin/python3.11 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

cnn:
	$(PY) src/train.py --model cnn --epochs 3

hybrid:
	$(PY) src/train.py --model hybrid --epochs 3 --timesteps 20

freeze:
	$(PIP) freeze | sed '/^-e /d' > requirements.txt

clean:
	rm -rf __pycache__ */__pycache__ .egg-info .pytest_cache .mypy_cache results/

fmt:
	@command -v black >/dev/null 2>&1 && black src || echo "black not installed"
	@command -v ruff >/dev/null 2>&1 && ruff check --fix src || echo "ruff not installed"

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check src || echo "ruff not installed"

spec:
	@command -v specify >/dev/null 2>&1 && specify init --here || echo "specify not found"
EOF

# 12) Initialize Spec-Kit docs
if (( CREATE_SPEC == 1 )); then
  if command -v specify >/dev/null 2>&1; then
    # Auto-confirm init if it prompts; safe to ignore failures
    specify init --here <<<'y' || true
  else
    echo "WARN: 'specify' not found; skip Spec-Kit init."
  fi
fi

# 13) Git init + first commit
if (( GIT_INIT == 1 )); then
  if [[ ! -d .git ]]; then
    git init
    git add .
    git commit -m "Bootstrap: spec-kit + venv + CNN–SNN skeleton" || true
    echo "Git repo initialized."
  else
    echo "Git already initialized."
  fi
fi

echo
echo "✅ Setup complete!"
echo "Next steps:"
echo " source ${VENV_DIR}/bin/activate"
echo " make cnn    # train baseline"
echo " make hybrid # train hybrid"

