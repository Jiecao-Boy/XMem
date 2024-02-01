import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image

from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar
from torchvision import transforms


try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')


torch.autograd.set_grad_enabled(False)

model_dir = './scripts/saves/XMem.pth'
mask_path = '/media/data/third_person_man/mask_test/Annotations/video1/00001.png'
image_directory = '/media/data/third_person_man/mask_test/JPEGImages/video1'
out_dir = '/media/data/third_person_man/box_open/test_mask'



class OnlineMask():
    def __init__(self, model_dir, mask_path, image_directory, out_dir):
        self.first_mask_loaded = False

        self.mask = Image.open(mask_path)
        # mask = Image.open(file_path).convert('P')
        self.mask = np.array(self.mask, dtype=np.uint8)

        # Get the list of files in the frames directory
        if image_directory is not None:
            self.image_directory = image_directory
            self.frame_files = sorted(os.listdir(image_directory))
        self.out_dir = out_dir

        config = {'model': './scripts/saves/XMem.pth', 'd16_path': '../DAVIS/2016', 'd17_path': '../DAVIS/2017', 'y18_path': '../YouTube2018', 'y19_path': '../YouTube', 'lv_path': '../long_video_set', 'generic_path': '/media/data/third_person_man/mask_test', 'dataset': 'G', 'split': 'val', 'output': '/media/data/third_person_man/mask_test/test', 'save_all': True, 'benchmark': False, 'disable_long_term': False, 'max_mid_term_frames': 10, 'min_mid_term_frames': 5, 'max_long_term_elements': 10000, 'num_prototypes': 128, 'top_k': 30, 'mem_every': 5, 'deep_update_every': -1, 'save_scores': False, 'flip': False, 'size': 480, 'enable_long_term': True, 'enable_long_term_count_usage': True}

        torch.autograd.set_grad_enabled(False)


        self.network = XMem(config, model_dir).cuda().eval()
        if model_dir is not None:
            model_weights = torch.load(model_dir)
            self.network.load_weights(model_weights, init_as_zero_if_needed=True)
        else:
            print('No model loaded.')
        self.mapper = MaskMapper()
        self.processor = InferenceCore(self.network, config=config)

        if self.mask is not None:
            self.first_mask_loaded = True

    def get_mask(self, frame_num, rgb, msk = None):
        if not self.first_mask_loaded:
            # no point to do anything without a mask
            return
        
        # with torch.cuda.amp.autocast(enabled=not args.benchmark):
        with torch.cuda.amp.autocast(enabled=True):
             # Map possibly non-continuous labels to continuous ones
            
            to_tensor = transforms.ToTensor()
            if msk is not None:
                # np.set_printoptions(threshold=np.inf)
                msk = torch.from_numpy(msk.astype(np.uint8))
                msk, labels = self.mapper.convert_mask(msk.numpy())
                msk = torch.Tensor(msk).cuda()
                msk = msk.unsqueeze(0)
                
                # if need resize: 
                h, w = msk.shape[-2:]
                min_hw = min(h, w)
                msk = F.interpolate(msk, (int(h/min_hw*480), int(w/min_hw*480)), 
                    mode='nearest')
                msk = msk[0]

                self.processor.set_all_labels(list(self.mapper.remappings.values()))
            else:
                labels = None

            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = to_tensor(rgb).cuda()

            print('image size:{}'.format(rgb.shape))
            if msk is not None:
                print('mask size: {}'.format(msk.shape))
            # Run the model on this frame
            prob = self.processor.step(rgb, msk, labels, end=False)

            # # Upsample to original size if needed
            # if need_resize:
            #     prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

            # end.record()
            # torch.cuda.synchronize()
            # total_process_time += (start.elapsed_time(end)/1000)
            # total_frames += 1
            # if args.flip:
            #     prob = torch.flip(prob, dims=[-1])

            # Probability mask -> index mask
            out_mask = torch.max(prob, dim=0).indices
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)


            # Save the mask
            os.makedirs(self.out_dir, exist_ok=True)
            out_mask = self.mapper.remap_index_mask(out_mask)
            out_img = Image.fromarray(out_mask)
            # if vid_reader.get_palette() is not None:
            #     out_img.putpalette(vid_reader.get_palette())
            out_img.save(os.path.join(self.out_dir, str(frame_num) + '.png'))

    def get_mask_batch(self):
        # Filter the list to include only PNG files
        frame = 0
        for frame_file in self.frame_files:
            # Construct the file paths for the frame and mask
            frame_path = os.path.join(self.image_directory, frame_file)
            # Load the frame and mask
            img = Image.open(frame_path).convert('RGB')
            shape = np.array(img).shape[:2]

            img = np.array(im_transform(img))

            if frame == 0:
                masker.get_mask(frame, img, self.mask)
            else:
                masker.get_mask(frame, img)
            frame += 1


            

        

        
if __name__ == '__main__':
    # preprare a initial mask

    # run image subscriber, get image
    im_normalization = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        im_normalization,
        transforms.Resize(480, interpolation=InterpolationMode.BILINEAR),
    ])


    masker = OnlineMask(model_dir, mask_path, image_directory, out_dir)
    masker.get_mask_batch()




    








