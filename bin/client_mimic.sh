#!/usr/bin/env bash

dir=`pwd`
parentdir="$(dirname "$dir")"
auto=$parentdir/auto_server
bandit=$parentdir/bandit
bayesian=$parentdir/bayesian_research

export PYTHONPATH=$bayesian:$PYTHONPATH
export PYTHONPATH=$auto:$PYTHONPATH
export PYTHONPATH=$bandit:$PYTHONPATH
export PYTHONPATH=$parentdir:$PYTHONPATH

python ../auto_server/client_mimic.py /home/chris/talkingdata/AutoML/a_data/a1a \
                                      /home/chris/talkingdata/AutoML/a_data/a2a