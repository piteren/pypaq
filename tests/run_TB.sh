#!/bin/bash
pkill tensorboard
nohup tensorboard --logdir="$PWD/_tmp/tbwr" >/dev/null 2>&1 &
