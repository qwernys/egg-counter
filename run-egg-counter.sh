docker run -d --restart unless-stopped \
    --gpus all \
    --name egg-counter \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    egg-counter python3 main-py