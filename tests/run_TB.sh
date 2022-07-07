#!/bin/bash
pkill tensorboard
nohup tensorboard --logdir="$PWD/_tmp" >/dev/null 2>&1 &
