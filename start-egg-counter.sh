#!/bin/bash

echo "Starting egg-counter container..."
# Allow local root to access X server
xhost +local:root
docker start egg-counter
echo "Egg-counter container started."
read -p "Press Enter to continue..."