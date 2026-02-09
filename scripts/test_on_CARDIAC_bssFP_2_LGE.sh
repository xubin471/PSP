#!/bin/bash
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1
###### Shared configs ######
SOURCE_DATASET='CARDIAC_bssFP'
TARGET_DATASET='CARDIAC_LGE'

NWORKER=0
RUNS=1
ALL_EV=(0 1 2 3 4) # 5-fold cross validation (0, 1, 1, 3, 4)
TEST_LABEL=[1,2,3]
#TEST_LABEL="'[1,2,3,4]'"
###### Training configs ######
NSTEP=50000
DECAY=0.98

MAX_ITER=5000 # defines the size of an epoch
SNAPSHOT_INTERVAL=1000 # interval for saving snapshot
SEED=2025

N_PART=3 # defines the number of chunks for evaluation
ALL_SUPP=(2) # CHAOST2: 0-4, CMR: 0-7
model_id=(50000)
#model_id=($(seq 39000 -1000 10000))
echo ========================================================================

for id in "${model_id[@]}"
do
  rm -rf results.txt
  rm -rf trial_data_analyze/trial_result
  for EVAL_FOLD in "${ALL_EV[@]}"
  do
     PREFIX="test_bssFP2LGE_cv${EVAL_FOLD}"
     echo $PREFIX
     LOGDIR="./result/${SOURCE_DATASET}_2_${TARGET_DATASET}"

      if [ ! -d $LOGDIR ]
      then
         mkdir -p $LOGDIR
      fi
      for SUPP_IDX in "${ALL_SUPP[@]}"
      do
         RELOAD_MODEL_PATH="exps_on_${SOURCE_DATASET}/CDFS_train_${SOURCE_DATASET}_cv${EVAL_FOLD}/1/snapshots/${id}.pth"
         echo "${RELOAD_MODEL_PATH}"
         python test.py with \
         mode="test" \
         dataset=$TARGET_DATASET \
         num_workers=$NWORKER \
         n_steps=$NSTEP \
         eval_fold=$EVAL_FOLD \
         max_iters_per_load=$MAX_ITER \
         supp_idx=$SUPP_IDX \
         test_label=$TEST_LABEL \
         seed=$SEED \
         n_part=$N_PART \
         reload_model_path=$RELOAD_MODEL_PATH \
         save_snapshot_every=$SNAPSHOT_INTERVAL \
         lr_step_gamma=$DECAY \
         path.log_dir=$LOGDIR
      done

  done
  grep -v '^$' results.txt | awk -v id=${id} '{sum+=$1;count++} END {printf "mean_dice: %.4f  epoch:%s \n", sum/count, id}' >>  ${LOGDIR}/mean_dice.txt
  grep -v '^$' results.txt | awk -v id=${id} '{sum+=$1;count++} END {printf "mean_dice: %.4f  epoch:%s \n", sum/count, id}'
done

