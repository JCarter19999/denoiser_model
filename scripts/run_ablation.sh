#!/usr/bin/env bash
set -euo pipefail

decorrupt-evaluate --config configs/evaluate.yaml --method none
decorrupt-evaluate --config configs/evaluate.yaml --method classical
decorrupt-evaluate --config configs/evaluate.yaml --method learned --checkpoint outputs/restoration/best.pt
decorrupt-evaluate --config configs/evaluate.yaml --method learned --checkpoint outputs/task_aware/best.pt
