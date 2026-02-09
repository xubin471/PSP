import os
import subprocess
import sys
print(sys.executable)

# 共享配置
GPUID1 = 1  # 使用的GPU
DATASET = 'SABS'  # 数据集
NWORKER = 0  # 线程数
RUNS = 1  # 运行次数
ALL_EV = [0,1,2,3,4]  # 5-fold cross validation (0, 1, 2, 3, 4)

TEST_LABEL = [1, 2, 3, 6]
EXCLUDE_LABEL = None
USE_GT = False

# 训练配置
NSTEP = 40000  # 训练的总步数
DECAY = 0.90  # 学习率的衰减系数
MAX_ITER = 1000  # 定义每个 epoch 的大小
SNAPSHOT_INTERVAL = 10000  # 保存快照的间隔
SEED = 2023
supp_idx = 2

model_id = 25000
# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID1)

# 创建日志目录
LOGDIR = f"./result"
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

# 循环每个fold进行训练
for EVAL_FOLD in ALL_EV:
    reload_model_path = f"BePMI_exps_on_CHAOST2/BePMI_train_CHAOST2_cv0/1/snapshots/{model_id}.pth"
    PREFIX = f"test_{DATASET}_cv{EVAL_FOLD}"


    # 训练命令的参数
    command = [
        '/home/cs4007/anaconda3/envs/xubin/bin/python', 'test.py', 'with',
        f"mode='test'",
        f"supp_idx={supp_idx}",
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
        f"path.log_dir={LOGDIR}",
        f"reload_model_path={reload_model_path}"
    ]

    # 执行训练命令
    subprocess.run(command)
