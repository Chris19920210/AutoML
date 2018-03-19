#!/bin/bash

dir=`pwd`
parentdir="$(dirname "$dir")"
auto=$parentdir/auto_server
bandit=$parentdir/bandit
bayesian=$parentdir/bayesian_research

export PYTHONPATH=$bayesian:$PYTHONPATH
export PYTHONPATH=$auto:$PYTHONPATH
export PYTHONPATH=$bandit:$PYTHONPATH
export PYTHONPATH=$parentdir:$PYTHONPATH

python ../auto_server/rpc_auto_server.py \
       --basic-config ../conf/config.properties \
       --processes 4 \
       --tune-times 10 \
       --bandit BanditEXP3