import os
import subprocess

# 共享配置
GPUID1 = 0  # 使用的GPU
DATASET = 'ABDOMEN_CT'  # 数据集
NWORKER = 0  # 线程数
RUNS = 1  # 运行次数
ALL_EV = [0]  # 5-fold cross validation (0, 3, 3, 3, 4)
TEST_LABEL = []
EXCLUDE_LABEL = None
USE_GT = False

# 训练配置
NSTEP = 50000  # 训练的总步数
DECAY = 0.98  # 学习率的衰减系数
MAX_ITER = 1000  # 定义每个 epoch 的大小
SNAPSHOT_INTERVAL = 1000  # 保存快照的间隔
SEED = 2023

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID1)

# 创建日志目录
LOGDIR = f"./exps_on_{DATASET}_multi_91"
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

# 循环每个fold进行训练
for EVAL_FOLD in ALL_EV:
    PREFIX = f"train_{DATASET}_cv{EVAL_FOLD}"
    print(f"Training for {PREFIX}")

    # 训练命令的参数
    command = [
        '/home/cs4007/anaconda3/envs/xubin/bin/python', 'train.py', 'with',
        f"mode='train'",
        f"dataset={DATASET}",
        f"num_workers={NWORKER}",
        f"n_steps={NSTEP}",
        f"eval_fold={EVAL_FOLD}",
        f"test_label={TEST_LABEL}",
        f"exclude_label={EXCLUDE_LABEL}",
        f"use_gt={USE_GT}",
        f"max_iters_per_load={MAX_ITER}",
        f"seed={SEED}",
        f"save_snapshot_every={SNAPSHOT_INTERVAL}",
        f"lr_step_gamma={DECAY}",
        f"path.log_dir={LOGDIR}"
    ]

    # 执行训练命令
    subprocess.run(command)
