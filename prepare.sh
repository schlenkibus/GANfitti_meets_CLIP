pip install --upgrade torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install --upgrade https://download.pytorch.org/whl/nightly/cu111/torch-1.11.0.dev20211012%2Bcu111-cp37-cp37m-linux_x86_64.whl https://download.pytorch.org/whl/nightly/cu111/torchvision-0.12.0.dev20211012%2Bcu111-cp37-cp37m-linux_x86_64.whl
git clone https://github.com/NVlabs/stylegan3
git clone https://github.com/openai/CLIP
git clone https://github.com/crowsonkb/esgd.git
pip install -e ./CLIP
pip install einops ninja
apt update
apt install ffmpeg -y