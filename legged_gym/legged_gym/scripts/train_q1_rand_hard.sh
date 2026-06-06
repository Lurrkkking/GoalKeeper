#!/usr/bin/env bash
set -e

cd /root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/legged_gym/scripts

python train.py \
    --task=q1_hard \
    --exptid=stand_urdf_5_014_rand_hard \
    --num_envs=1200 \
    --max_iterations=10000
