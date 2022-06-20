from subprocess import DEVNULL
import sys
sys.path.append('./CLIP')
sys.path.append('./stylegan3')

import os, time, glob
import pickle
import numpy as np
import torch
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import PIL
import argparse
import boilerplate

parser = argparse.ArgumentParser(description='Create Graffiti')
parser.add_argument(
    '--outputpath',
    type=str,
    default='/data/GANfitti/generated_clip/',
    help='Path to output directory'
)

parser.add_argument(
    '--prompt',
    type=str,
    default='a sick graffiti',
    help='Prompt Text'
)

parser.add_argument(
    '--seed',
    type=int,
    default=-1,
    help='Seed for random number generator'
)

parser.add_argument(
    '--iterations',
    type=int,
    default=100,
    help='Number of iterations'
)

parser.add_argument(
    '--save_video',
    type=bool,
    default=False,
    help='Save video'
)

parser.add_argument(
    '--network_path',
    type=str,
    default='./network-snapshot-000075.pkl',
    help='Path to network snapshot'
)

parser.add_argument(
    '--video_length',
    type=int,
    default=20,
    help='Length of video in seconds'
)
    

args = parser.parse_args()



device = torch.device('cuda:0')
print('Using device:', device, file=sys.stderr)




##START

create_video = args.save_video
seed = args.seed
iterations = args.iterations
prompt = args.prompt
network_path = args.network_path
output_path = args.outputpath


clip_model = boilerplate.CLIP(device)
model_path = args.network_path

with open(model_path, 'rb') as fp:
  G = pickle.load(fp)['G_ema'].to(device)

zs = torch.randn([10000, G.mapping.z_dim], device=device)
w_stds = G.mapping(zs, None).std(0)

texts = prompt
steps = iterations
seed = args.seed

if seed == -1:
    seed = np.random.randint(0,9e9)
    print(f"Your random seed is: {seed}")

texts = [frase.strip() for frase in texts.split("|") if frase]

targets = [clip_model.embed_text(text) for text in texts]

#@markdown #**Run the model** 🚀

# Actually do the run

print(seed)

tf = Compose([
  Resize(224),
  lambda x: torch.clamp((x+1)/2,min=0,max=1),
  ])

def run(timestring, seed):
  
  if seed == -2:
    np.random.seed(-1)
    torch.manual_seed(np.random.random_integers(0,9e9))
  else:
    torch.manual_seed(seed)
    
  # Init
  # Sample 32 inits and choose the one closest to prompt

  with torch.no_grad():
    qs = []
    losses = []
    for _ in range(8):
      q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
      images = G.synthesis(q * w_stds + G.mapping.w_avg)
      embeds = boilerplate.embed_image(images.add(1).div(2), clip_model)
      loss = boilerplate.prompts_dist_loss(embeds, targets, boilerplate.spherical_dist_loss).mean(0)
      i = torch.argmin(loss)
      qs.append(q[i])
      losses.append(loss[i])
    qs = torch.stack(qs)
    losses = torch.stack(losses)
    # print(losses)
    # print(losses.shape, qs.shape)
    i = torch.argmin(losses)
    q = qs[i].unsqueeze(0).requires_grad_()

  # Sampling loop
  q_ema = q
  opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0,0.999))
  loop = range(steps)
  for i in loop:
    opt.zero_grad()
    w = q * w_stds
    image = G.synthesis(w + G.mapping.w_avg, noise_mode='const')
    embed = boilerplate.embed_image(image.add(1).div(2), clip_model)
    loss = boilerplate.prompts_dist_loss(embed, targets, boilerplate.spherical_dist_loss).mean()
    loss.backward()
    opt.step()

    q_ema = q_ema * 0.9 + q * 0.1
    image = G.synthesis(q_ema * w_stds + G.mapping.w_avg, noise_mode='const')

    if i % 10 == 0:
      #TODO change to notify flask and send the image
      #display(TF.to_pil_image(tf(image)[0]))
      print(f"Image {i}/{steps} | Current loss: {loss}")
    pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
    os.makedirs(f'samples/{timestring}', exist_ok=True)
    pil_image.save(f'samples/{timestring}/{i:04}.jpg')

num_seeds = 10

try:
  if seed == -2:
    #create random seeds 
    seeds = np.random.randint(0,9e9,num_seeds)
  else:
    seeds = [seed]
  for s in seeds:
    timestring = time.strftime('%Y%m%d%H%M%S')
    run(timestring, s)

    archive_name = "optional"

    archive_name = boilerplate.slugify(archive_name)

    if archive_name != "optional":
      fname = archive_name
      # os.rename(f'samples/{timestring}', f'samples/{fname}')
    else:
      fname = timestring

    # Save images as a tar archive
    #!tar cf samples/{fname}.tar samples/{timestring}

    #if os.path.isdir('drive/MyDrive/samples'):
    #  shutil.copyfile(f'samples/{fname}.tar', f'drive/MyDrive/samples/{fname}.tar')
    #else:
    #  files.download(f'samples/{fname}.tar')

    ## create video

    if create_video:
      frames = os.listdir(f"samples/{timestring}")
      frames = len(list(filter(lambda filename: filename.endswith(".jpg"), frames))) #Get number of jpg generated

      init_frame = 1 #This is the frame where the video will start
      last_frame = frames #You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

      min_fps = 10
      max_fps = 60

      total_frames = last_frame-init_frame

      #Desired video time in seconds
      #@param {type:"number"}
      video_length = args.video_length
      #Video filename
      #@param {type:"string"}
      video_name = prompt + "-" + str(s) 
      video_name = boilerplate.slugify(video_name)

      if not video_name:
          video_name = "video"
      
      frames = []
      print('Generating video...')
      for i in range(init_frame,last_frame): #
          filename = f"samples/{timestring}/{i:04}.jpg"
          frames.append(PIL.Image.open(filename))

      fps = np.clip(total_frames/video_length,min_fps,max_fps)

      print("Generating video...")

      from subprocess import Popen, PIPE, DEVNULL
      p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', f'samples/{video_name}.mp4'], stdin=PIPE, stdout=DEVNULL)
      
      for im in frames:
          im.save(p.stdin, 'PNG')
      
      p.stdin.close()

      #flush p
      #p.communicate()
      p.wait()


      print("The video is ready")

except KeyboardInterrupt:
  pass


#download images
#@param {type:"string"}
