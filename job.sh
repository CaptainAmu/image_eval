#!/bin/bash
#SBATCH -N 1                          # 申请1个节点
#SBATCH --job-name=image_eval
#SBATCH --output=logs/%j_out.txt
#SBATCH --error=logs/%j_err.txt
#SBATCH --time=00:20:00
#SBATCH --mem=36000                   # 36G 内存
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --partition=normal

# 防止 .local 库干扰
export PYTHONNOUSERSITE=1

# 进入提交作业时的目录，保证 images/、results.csv 等相对路径正确
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# -u 无缓冲输出，便于实时看日志
srun -u /slurm-storage/shucli/.conda/envs/image_eval/bin/python -u compute_image_metrics.py \
  --images_dir output/images \
  --csv_path output/results.csv \
  --output output/metrics_results.csv