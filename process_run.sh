#!/bin/bash

RUN=$1
DARKRUN=$2
XGMLOWER=$3
XGMUPPER=$4

DIR="./slurm_log"

if [ ! -d $DIR ]; then
    mkdir $DIR
fi

for MODULE in {0..15..1}; do
    sbatch -p upex -t 100 -o "${DIR}/slurm-%A.out" ./process_module.sh $RUN $MODULE $DARKRUN $XGMLOWER $XGMUPPER
done
