import torch
from torch.utils.data import Dataset
import numpy as np
import imageio
import warnings
from PIL import Image
from copy import deepcopy
from .utils import load_meta_data, get_rays, extract_patches


class RINDataset(Dataset):
    """ Ray Image Normal Dataset """

    def __init__(self, args, mode='train', n_views=None):
        self.args = args
        images, c2w, H, W, focal_x, focal_y, image_paths, bg_masks = load_meta_data(args, mode=mode)
        num_imgs = len(image_paths)

        self.num_imgs = num_imgs
        coord_scale = args.coord_scale
        if coord_scale != 1:
            scaling_matrix = torch.tensor([[coord_scale, 0, 0, 0],
                                           [0, coord_scale, 0, 0],
                                           [0, 0, coord_scale, 0],
                                           [0, 0, 0, 1]], dtype=torch.float32)
            c2w = torch.matmul(scaling_matrix, c2w)
        print("c2w: ", c2w.shape)

        self.H = H
        self.W = W
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.c2w = c2w      # (N, 4, 4)
        self.image_paths = image_paths
        self.images = images    # (N, H, W, C) or None
        self.bg_masks = bg_masks    # (N, H, W) or None

        if mode=='train' and n_views:
            if n_views > num_imgs:
                warnings.warn(f'n_views is greater than the total number of training images. n_veiw is set to {num_imgs} and all the views are selected')
            
            sel_indices = torch.randperm(num_imgs)[:n_views]
            self.image_paths = [image_paths[i] for i in sel_indices]
            self.images = images[sel_indices] if args.read_offline else images  # (n_views, H, W, C) or None
            self.bg_masks = bg_masks[sel_indices] if args.read_offline else bg_masks  # (n_views, H, W, C) or None
            self.num_imgs = n_views
            self.c2w = c2w[sel_indices]      # (n_views, 4, 4)
            

        if self.args.read_offline:
            rays_o, rays_d = get_rays(H, W, focal_x, focal_y, self.c2w)
            self.rayd = rays_d      # (N, H, W, 3)
            self.rayo = rays_o      # (N, 3)

        if self.args.extract_patch == True and self.args.extract_online == False and self.args.read_offline == True:
            img_patches, rayd_patches, rayo_patches, num_patches, bg_mask_patches = extract_patches(self.images, rays_o, rays_d, args, self.bg_masks)
            # (N, n_patches, patch_height, patch_width, C) or None
            self.img_patches = img_patches
            # (N, n_patches, patch_height, patch_width, 3)
            self.rayd_patches = rayd_patches
            self.rayo_patches = rayo_patches    # (N, n_patches, 3)
            self.num_patches = num_patches
            self.bg_mask_patches = bg_mask_patches

    def _read_image_from_path(self, image_idx):
        image_path = self.image_paths[image_idx]
        image = imageio.imread(image_path)
        image = Image.fromarray(image).resize((self.W, self.H))
        image = (np.array(image) / 255.).astype(np.float32)
        bg_mask = np.ones_like(image[..., 0])    # 0 if background, 1 otherwise

        if self.args.white_bg and image.shape[-1] == 4:
            bg_mask = image[..., 3]
            image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
        elif not self.args.white_bg:
            image = image[..., :3]
            mask = image.sum(-1) == 3.0
            image[mask] = 0.
            bg_mask[mask] = 0.  #FIXME NOT tested.

        image = torch.from_numpy(image).float()
        bg_mask = torch.from_numpy(bg_mask).float()

        rayo, rayd = get_rays(self.H, self.W, self.focal_x, self.focal_y, self.c2w[image_idx:image_idx+1])

        return image, rayo, rayd, bg_mask

    def __len__(self):
        if self.args.extract_patch == True and self.args.extract_online == False and self.args.read_offline == True:
            return self.num_imgs * self.num_patches
        else:
            return self.num_imgs

    def __getitem__(self, idx):
        if self.args.extract_patch == True and self.args.extract_online == False and self.args.read_offline == True:
            img_idx = idx // self.num_patches
            patch_idx = idx % self.num_patches
            return img_idx, patch_idx, \
                self.img_patches[img_idx, patch_idx] if self.img_patches is not None else 0, \
                self.rayd_patches[img_idx, patch_idx], \
                self.rayo_patches[img_idx, patch_idx]

        elif self.args.extract_patch == True and self.args.extract_online == True:
            img_idx = idx
            args = self.args
            # args.patches.max_patches = 1          # Important: the DictAsMember type does not update the values like this (This was a bug in the main code)
            args['patches']['max_patches'] = 1
            if self.args.read_offline:
                img_patches, rayd_patches, rayo_patches, bg_mask_patches, _ = extract_patches(self.images[img_idx:img_idx+1],
                                                                                self.rayo[img_idx:img_idx+1],
                                                                                self.rayd[img_idx:img_idx+1],
                                                                                args)
            else:
                image, rayo, rayd, bg_mask = self._read_image_from_path(img_idx)
                img_patches, rayd_patches, rayo_patches, bg_mask_patches, _ = extract_patches(image[None, ...], rayo, rayd, args, bg_mask[None, ...])

            return img_idx, 0, \
                img_patches[0, 0] if img_patches is not None else 0, \
                rayd_patches[0, 0], \
                rayo_patches[0, 0], \
                bg_mask_patches[0, 0]
        else:
            if self.args.read_offline:
                return idx, 0, self.images[idx] if self.images is not None else 0, \
                    self.rayd[idx], self.rayo[idx], self.bg_mask_patches[idx]
            else:
                image, rayo, rayd, bg_mask = self._read_image_from_path(idx)
                return idx, 0, image, rayd.squeeze(0), rayo.squeeze(0), bg_mask

    def get_full_img(self, img_idx):
        if self.args.read_offline:
            return self.images[img_idx].unsqueeze(0) if self.images is not None else None, \
                self.rayd[img_idx].unsqueeze(0), self.rayo[img_idx].unsqueeze(0)
        else:
            image, rayo, rayd, bg_mask = self._read_image_from_path(img_idx)
            return image[None, ...], rayd, rayo, bg_mask[None, ...]

    def get_c2w(self, img_idx):
        return self.c2w[img_idx]

    def get_new_rays(self, c2w):
        return get_rays(self.H, self.W, self.focal_x, self.focal_y, c2w)
