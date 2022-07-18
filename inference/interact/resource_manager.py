import os
from os import path

from functools import lru_cache
import shutil

import cv2
from PIL import Image
import numpy as np

from util.palette import davis_palette
import progressbar


class ResourceManager:
    def __init__(self, config):
        # determine inputs
        images = config['images']
        video = config['video']
        self.workspace = config['workspace']
        self.size = config['size']
        self.palette = davis_palette

        # create temporary workspace if not specified
        if self.workspace is None:
            if images is not None:
                basename = path.basename(images)
            elif video is not None:
                basename = path.basename(video)[:-4]
            else:
                raise NotImplementedError(
                    'Either images, video, or workspace has to be specified')

            self.workspace = path.join('./workspace', basename)

        print(f'Workspace is in: {self.workspace}')

        # determine the location of input images
        need_decoding = False
        need_resizing = False
        if path.exists(path.join(self.workspace, 'images')):
            pass
        elif images is not None:
            need_resizing = True
        elif video is not None:
            # will decode video into frames later
            need_decoding = True

        # create workspace subdirectories
        self.image_dir = path.join(self.workspace, 'images')
        self.mask_dir = path.join(self.workspace, 'masks')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)

        # convert read functions to be buffered
        self.get_image = lru_cache(maxsize=config['buffer_size'])(self._get_image_unbuffered)
        # The check itself should not be buffered
        self._get_mask_buffered_without_check = lru_cache(maxsize=config['buffer_size'])(
                                    self._get_mask_unbuffered_without_check)

        # extract frames from video
        if need_decoding:
            self._extract_frames(video)

        # copy/resize existing images to the workspace
        if need_resizing:
            self._copy_resize_frames(images)

        # read all frame names
        self.names = sorted(os.listdir(self.image_dir))
        self.names = [f[:-4] for f in self.names] # remove extensions
        self.length = len(self.names)

        assert self.length > 0, f'No images found! Check {self.workspace}/images. Remove folder if necessary.'

        print(f'{self.length} images found.')

        self.height, self.width = self.get_image(0).shape[:2]

    def _extract_frames(self, video):
        cap = cv2.VideoCapture(video)
        frame_index = 0
        print(f'Extracting frames from {video} into {self.image_dir}...')
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        while(cap.isOpened()):
            _, frame = cap.read()
            if frame is None:
                break
            if self.size > 0:
                h, w = frame.shape[:2]
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
            cv2.imwrite(path.join(self.image_dir, f'{frame_index:07d}.jpg'), frame)
            frame_index += 1
            bar.update(frame_index)
        bar.finish()
        print('Done!')

    def _copy_resize_frames(self, images):
        image_list = os.listdir(images)
        print(f'Copying/resizing frames into {self.image_dir}...')
        for image_name in progressbar.progressbar(image_list):
            if self.size < 0:
                # just copy
                shutil.copy2(path.join(images, image_name), self.image_dir)
            else:
                frame = cv2.imread(path.join(images, image_name))
                h, w = frame.shape[:2]
                new_w = (w*self.size//min(w, h))
                new_h = (h*self.size//min(w, h))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame,dsize=(new_w,new_h),interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, image_name), frame)
        print('Done!')

    def save_mask(self, ti, mask):
        # mask should be uint8 H*W without channels
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)

        mask = Image.fromarray(mask)
        mask.putpalette(self.palette)
        mask.save(path.join(self.mask_dir, self.names[ti]+'.png'))

    def _get_image_unbuffered(self, ti):
        # returns H*W*3 uint8 array
        assert 0 <= ti < self.length

        image = Image.open(path.join(self.image_dir, self.names[ti]+'.jpg'))
        image = np.array(image)
        return image

    def _get_mask_unbuffered_without_check(self, ti):
        # returns H*W uint8 array
        mask = Image.open(path.join(self.mask_dir, self.names[ti]+'.png'))
        mask = np.array(mask)
        return mask

    def get_mask(self, ti):
        # returns H*W uint8 array
        assert 0 <= ti < self.length

        mask_path = path.join(self.mask_dir, self.names[ti]+'.png')
        if path.exists(mask_path):
            return self._get_mask_buffered_without_check(ti)
        else:
            # no mask there
            return None

    def __len__(self):
        return self.length

    @property
    def h(self):
        return self.height

    @property
    def w(self):
        return self.width
