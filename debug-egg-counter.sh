#!/bin/bash

echo "Debug mode started for egg-counter container."

# Allow local root to access X server
xhost +local:root

# Run docker container with debug mode and GPU access
docker run --rm --gpus all \
    --name egg-counter-debug \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    egg-counter python3 main.py --debug 

docker stop egg-counter-debug
docker rm egg-counter-debug

echo "Debug mode finished for egg-counter container."
read -p "Press Enter to continue..."