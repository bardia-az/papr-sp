import torch
import os
import numpy as np

from depth_prior.lib.models.multi_depth_model_auxiv2 import RelDepthModel_cIMLE
from depth_prior.lib.utils.net_tools import strip_prefix_if_present
from depth_prior.tools.utils import load_mean_var_adain



def compute_space_carving_loss(pred_depth:torch.Tensor, target_hypothesis:torch.Tensor, mask=None, norm_p=2, threshold=0.0):
    _, H, W = pred_depth.shape
    num_hypothesis = target_hypothesis.shape[0]

    pred_depth_repeated = pred_depth.unsqueeze(0).repeat(num_hypothesis, 1, 1, 1)

    ## L2 distance
    distances = torch.norm(pred_depth_repeated - target_hypothesis, p=norm_p, dim=(2, 3))

    if mask is not None:    #FIXME has not been checked
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:   #FIXME has not been checked
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)

    sample_min = torch.min(distances, axis=0)[0]
    sample_min_normalized = torch.pow(sample_min, norm_p) / (H * W) 
    loss = torch.mean(sample_min_normalized)

    return loss



def load_depth_prior_model(ckpt_dir:str, ckpt:str, d_latent:int = 32, ada_version:str = 'v2', device='cuda'):
    model = RelDepthModel_cIMLE(d_latent=d_latent, version=ada_version)
    model.to(device)

    ## freez the weights of the model as we only need it for inference
    for param in model.parameters():
        param.requires_grad = False

    ### Load model
    model_dict = model.state_dict()

    ckpt_file = os.path.join(ckpt_dir, ckpt)

    if os.path.isfile(ckpt_file):
        print("loading checkpoint %s" % ckpt_file)
        checkpoint = torch.load(ckpt_file)

        checkpoint['model_state_dict'] = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")
        depth_keys = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict} ## <--- some missing keys in the loaded model from the given model
        print(len(depth_keys))

        # Overwrite entries in the existing state dict
        model_dict.update(depth_keys)        

        # Load the new state dict
        model.load_state_dict(model_dict)

        print("Model loaded.")

    else:
        print("Error: Model does not exist.")
        exit()

    mean0, var0, mean1, var1, mean2, var2, mean3, var3 = load_mean_var_adain(os.path.join(ckpt_dir, "mean_var_adain.npy"), torch.device("cuda"))
    model.set_mean_var_shifts(mean0, var0, mean1, var1, mean2, var2, mean3, var3)
    print("Initialized adain mean and var.")

    return model



def get_depth_priors(model:RelDepthModel_cIMLE, img: torch.Tensor, sample_num:int = 20, d_latent:int = 32, rescaled:bool = False, device='cuda'):
    batch_size, H, W, C = img.shape
    img = img.permute(0, 3, 1, 2).to(device)

    ### Repeat for the number of samples
    num_images = img.shape[0]
    img = img.unsqueeze(1).repeat(1,sample_num, 1, 1, 1)
    img = img.view(-1, C, H, W)

    # rgb = torch.clone(data[0]).permute(1, 2, 0).to("cpu").detach().numpy() 
    # rgb = rgb[:, :, ::-1] ## dataloader is bgr
    # rgb = 255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())
    # rgb = np.array(rgb, np.int)

    ## Hard coded d_latent
    z = torch.normal(0.0, 1.0, size=(num_images, sample_num, d_latent))
    z = z.view(-1, d_latent).to(device)

    data = {}
    data['rgb'] = img
    pred_depth = model.inference(data, z, rescaled=rescaled)

    return pred_depth


