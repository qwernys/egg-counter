#!/bin/bash

# Allow local root to access X server
xhost +local:root

# Run docker container with debug mode and GPU access
docker run --rm --gpus all \
    --name egg-counter-debug \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    egg-counter python3 main-py --debug 

echo "Debug mode started for egg-counter container."
echo "To stop the debug container, run: docker stop egg-counter-debug"
echo "To remove the debug container, run: docker rm egg-counter-debug"
read -p "Press Enter to continue..."