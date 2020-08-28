#!/bin/bash

RUN=$1
MODULE=$2

source /usr/share/Modules/init/bash
module load exfel
module load exfel_anaconda3/1.1
python process_module.py --run-number $RUN --module $MODULE
