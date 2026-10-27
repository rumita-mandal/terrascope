#!/usr/bin/env bash
# Full training pipeline (CPU-friendly). ~35 min on 2 cores.
set -e
cd "$(dirname "$0")"
python3 -m terrascope.core.train pretrain --steps ${PRETRAIN_STEPS:-3000}
python3 -m terrascope.core.train finetune --steps ${FT_STEPS:-500} --labels ${LABELS:-100}
python3 -m terrascope.core.train scratch  --steps ${FT_STEPS:-500} --labels ${LABELS:-100}
LABELS=${LABELS:-100} python3 -m terrascope.core.evaluate
