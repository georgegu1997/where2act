set -x
set -u
set -e

CUDA_VISIBLE_DEVICES=1 python gen_offline_data.py \
  --data_dir ../data/gt_data-train_10cats_train_data-pulling \
  --data_fn ../stats/train_10cats_train_data_list.txt \
  --primact_types pulling \
  --num_processes 64 \
  --num_epochs 150 \
  --ins_cnt_fn ../stats/ins_cnt_15cats.txt

CUDA_VISIBLE_DEVICES=1 python gen_offline_data.py \
  --data_dir ../data/gt_data-train_10cats_test_data-pulling \
  --data_fn ../stats/train_10cats_test_data_list.txt \
  --primact_types pulling \
  --num_processes 64 \
  --num_epochs 10 \
  --ins_cnt_fn ../stats/ins_cnt_15cats.txt

CUDA_VISIBLE_DEVICES=1 python gen_offline_data.py \
  --data_dir ../data/gt_data-test_5cats-pulling \
  --data_fn ../stats/test_5cats_data_list.txt \
  --primact_types pulling \
  --num_processes 64 \
  --num_epochs 10 \
  --ins_cnt_fn ../stats/ins_cnt_5cats.txt

