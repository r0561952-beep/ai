"""Обучение средней модели (~45 млн параметров) на видеокарте Kaggle.

Запускается на Kaggle как фоновая задача (kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4)
или вручную: новый ноутбук → GPU T4 + Internet On → вставить этот файл в ячейку → Save & Run All.

Результат (вкладка Output):
  model_medium.pt — веса для чата
  train.log       — журнал обучения
  ckpt/full.pt    — полное состояние для продолжения обучения
"""
import os
import subprocess

BRANCH = "ccr-d2dd0252-6ksj1l"
REPO = "https://github.com/r0561952-beep/ai.git"
# значения можно переопределить переменными окружения (для пробного запуска без видеокарты)
WIKI_FILES = int(os.environ.get("WIKI_FILES", 10))           # частей Википедии из 21 (~450 тыс. статей)
TRAIN_MINUTES = float(os.environ.get("TRAIN_MINUTES", 420))  # 7 часов (лимит Kaggle — 12 ч на запуск)
PRESET = os.environ.get("PRESET", "medium")
DEVICE = os.environ.get("DEVICE", "cuda")
OUT = os.environ.get("OUT", "/kaggle/working")
WORK = os.environ.get("WORK", "/tmp/ai")


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


run(f"rm -rf {WORK} && git clone -q -b {BRANCH} {REPO} {WORK} && git -C {WORK} log --oneline -1")
os.chdir(WORK)
run("pip install -q tokenizers")
if DEVICE == "cuda":
    run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run(f"python prepare_wiki.py --files {WIKI_FILES}")
run(
    f"set -o pipefail; python train.py --preset {PRESET} --device {DEVICE} --mix wiki=0.8,qa=0.2 "
    "--batch_size 32 --lr 6e-4 --min_lr 6e-5 --warmup 500 "
    f"--max_iters 100000000 --max_minutes {TRAIN_MINUTES} --time_schedule --eval_every 1000 "
    f"--data_dir {WORK}/bins --ckpt_dir {OUT}/ckpt --out {OUT}/model_medium.pt "
    f"2>&1 | tee {OUT}/train.log"
)
run(f"python chat.py --model {OUT}/model_medium.pt -q 'Что такое Солнце?'")
run(f"python chat.py --model {OUT}/model_medium.pt -q 'Как приготовить блины?'")
