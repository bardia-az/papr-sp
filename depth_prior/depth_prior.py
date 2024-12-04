import torch
import os
import numpy as np

from depth_prior.lib.models.multi_depth_model_auxiv2 import RelDepthModel_cIMLE
from depth_prior.lib.utils.net_tools import strip_prefix_if_present
from depth_prior.tools.utils import load_mean_var_adain



def compute_space_carving_loss(pred_depth, target_hypothesis, is_joint=False, mask=None, norm_p=2, threshold=0.0):
    n_rays, n_points = pred_depth.shape
    num_hypothesis = target_hypothesis.shape[0]

    if target_hypothesis.shape[-1] == 1:
        ### In the case where there is no caching of quantiles
        target_hypothesis_repeated = target_hypothesis.repeat(1, 1, n_points)
    else:
        ### Each quantile here already picked a hypothesis
        target_hypothesis_repeated = target_hypothesis

    ## L2 distance
    # distances = torch.sqrt((pred_depth - target_hypothesis_repeated)**2)
    distances = torch.norm(pred_depth.unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=norm_p, dim=-1)

    if mask is not None:
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)

    if is_joint:
        ### Take the mean for all points on all rays, hypothesis is chosen per image
        quantile_mean = torch.mean(distances, axis=1) ## mean for each quantile, averaged across all rays
        samples_min = torch.min(quantile_mean, axis=0)[0]
        loss =  torch.mean(samples_min, axis=-1)
    else:
        ### Each ray selects a hypothesis
        best_hyp = torch.min(distances, dim=0)[0]   ## for each sample pick a hypothesis
        ray_mean = torch.mean(best_hyp, dim=-1) ## average across samples
        loss = torch.mean(ray_mean)  

    return loss




def load_depth_prior_model(ckpt_dir:str, ckpt:str, d_latent:int = 32, ada_version:str = 'v2', device='cuda'):
    model = RelDepthModel_cIMLE(d_latent=d_latent, version=ada_version)
    model.to(device)

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



def get_depth_priors(model:RelDepthModel_cIMLE, data, sample_num:int = 20, d_latent:int = 32, rescaled:bool = False, device='cuda'):
    batch_size = data.shape[0]
    C = data.shape[1]
    H = data.shape[2]
    W = data.shape[3]

    ### Repeat for the number of samples
    num_images = data.shape[0]
    data = data.unsqueeze(1).repeat(1,sample_num, 1, 1, 1)
    data = data.view(-1, C, H, W)

    # rgb = torch.clone(data[0]).permute(1, 2, 0).to("cpu").detach().numpy() 
    # rgb = rgb[:, :, ::-1] ## dataloader is bgr
    # rgb = 255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())
    # rgb = np.array(rgb, np.int)

    ## Hard coded d_latent
    z = torch.normal(0.0, 1.0, size=(num_images, sample_num, d_latent))
    z = z.view(-1, d_latent).cuda()

    pred_depth = model.inference(data, z, rescaled=rescaled)

    return pred_depth


