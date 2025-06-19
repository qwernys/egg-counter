# egg-counter
Download Debian 12
Boot in to debian os
If locked in black screen -> In GRUB boot menu  press **e** to edit boot parameters on debian boot. Find the line that starts with linux /boot/vmlinuz... and add nomodeset to the end of the line.

# Add user to sudoers group
su -
root@yourhostname:~#
usermod -aG sudo yourusername
reboot

# Download GPU Drivers
sudo apt install nvidia-driver firmware-misc-nonfree
sudo apt-get install -y nvidia-container-toolkit

# Install git
sudo apt-get update
sudo apt-get install git

# Download and start docker
sudo apt-get install docker.io -y
sudo systemctl enable docker
sudo systemctl start docker

git clone https://github.com/qwernys/egg-counter.git
cd egg-counter
docker build -t egg-counter .
xhost +local:root
docker run -d --restart unless-stopped --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix egg-counter --name egg-counter egg-counter python3 main.py

# Debug
docker run --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix egg-counter python3 main.py --debug
