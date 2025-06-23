#!/bin/bash

docker run -d --restart unless-stopped \
    --gpus all \
    --name egg-counter \
    --network=host \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    egg-counter python3 main.py