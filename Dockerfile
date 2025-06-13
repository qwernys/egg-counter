FROM nvidia/cuda:12.1.1-runtime-debian11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg libsm6 libxext6 git curl \
    && apt-get clean

RUN pip install --upgrade pip

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]