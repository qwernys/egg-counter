FROM nvidia/cuda:12.1.1-runtime-debian11

# Set working directory
WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python and pip to python3 if needed
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA 12.1 support first
RUN pip install --upgrade pip
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Copy and install project Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install ByteTrack
RUN git clone https://github.com/ifzhang/ByteTrack.git /app/ByteTrack \
    && pip install -r /app/ByteTrack/requirements.txt \
    && cd /app/ByteTrack && python setup.py develop

# Copy the rest of your project files
COPY . .

# Set default command
CMD ["python", "main.py"]