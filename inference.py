from .options.train_options import TrainOptions
import os
from .models import create_model
import torch
import numpy as np
import PIL.Image as img
from PIL import Image

def from_pretrained(base_path: str, model_name: str):
    opt = TrainOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.checkpoints_dir = base_path
    opt.name = model_name
    opt.phase = 'test'
    opt.isTrain = False
    if torch.cuda.is_available():
        opt.gpu_ids = [0]
    else:
        opt.gpu_ids = []
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    return model
    

def call(model, data):
    print("Calling model with data shape", data.shape)
    input_data = {'A': data[:1, :1, :256, :256], 'B': data[:1, :1, :256, :256], 'A_paths': [''], 'B_paths': ['']}
    model.set_input(input_data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    to_return = visuals['fake_B']
    print("Returning data shape", to_return.shape)
    return to_return

if __name__ == '__main__':
    path = '/home/otrolie/.config/blender/3.6/scripts/addons/terrain-ml/models/gans/sketch-to-planet'
    model = from_pretrained(path, 'Zoom 0')
    sketch_dir = '/home/otrolie/.config/blender/3.6/scripts/addons/terrain-ml/app/sketches/512x256.png'
    sketch = img.open(sketch_dir)
    sketch = np.array(sketch)[:256, :256, 0:1]
    img.fromarray(sketch[:, :, 0]).show()
    sketch = sketch.transpose((2, 0, 1)).astype(np.float32)
    sketch = sketch / 255
    sketch = sketch * 2 - 1
    sketch = torch.from_numpy(sketch)
    data = sketch.unsqueeze(0)
    output = call(model, data)
    im = output[0].cpu().numpy()
    im = np.transpose(im, (1, 2, 0))
    im = (im + 1) / 2
    im = (im * 255).astype(np.uint8)
    im = Image.fromarray(im[:, :, 0])
    im.show()

