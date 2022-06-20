FROM stylegan3:latest

ENV TZ=Europe/Berlin

RUN pip install --upgrade torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/NVlabs/stylegan3
RUN git clone https://github.com/openai/CLIP
RUN git clone https://github.com/crowsonkb/esgd.git
RUN pip install -e ./CLIP
RUN pip install einops ninja
RUN apt update -y
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg --no-install-recommends --no-install-suggests -y