#!/bin/bash

RUN=$1
MODULE=$2
DARKRUN=$3
XGMLOWER=$4
XGMUPPER=$5

source /usr/share/Modules/init/bash
module load exfel
module load exfel_anaconda3/1.1
python process_module.py --run-number $RUN --module $MODULE --dark-run $DARKRUN --xgm-lower $XGMLOWER --xgm-upper $XGMUPPER
