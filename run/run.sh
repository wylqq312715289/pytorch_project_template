#!/usr/bin/env bash

MAIN_ROOT=/disk2/wangyulong/SpeechSegmentation  # 需定制

############# mkdir path if not exist ########
if [ ! -d ${MAIN_ROOT}/cache ]; then mkdir -p ${MAIN_ROOT}/cache; fi # 需定制
if [ ! -d ${MAIN_ROOT}/logs ]; then mkdir -p ${MAIN_ROOT}/logs; fi # 需定制

############# update path ##############
export PATH=$PATH:${MAIN_ROOT}/src/utils
export PATH=$PATH:${MAIN_ROOT}/src/shell


############# update python path ##############
export PYTHONPATH=${PYTHONPATH}:$MAIN_ROOT/src/bin
export PYTHONPATH=${PYTHONPATH}:$MAIN_ROOT/src/utils
export PYTHONPATH=${PYTHONPATH}:$MAIN_ROOT/src/models
export PYTHONPATH=${PYTHONPATH}:$MAIN_ROOT/src/solver



python ${MAIN_ROOT}/src/bin/main.py --conf_file ${MAIN_ROOT}/conf/wyl.yaml


