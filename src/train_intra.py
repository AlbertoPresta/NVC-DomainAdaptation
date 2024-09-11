import os



from models.image_model import DMCI


import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import argparse
import math
import random
import torch


from torchvision import transforms
 
from torch.utils.data import Dataset
import os
from PIL import Image




import torch
import math
from pytorch_msssim import ms_ssim

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255):
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean()) #ddd


def compute_metrics(org, rec, max_val = 255):
    metrics = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics

class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path =  [os.path.join(data_dir,f) for f in os.listdir(data_dir)] #sorted(glob(os.path.join(self.data_dir, "*.*")))

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        
        return self.transform(image)
    

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")


    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )


    parser.add_argument(
        "--num_images",
        type=int,
        default=300000,
        help="Dataloaders threads (default: %(default)s)",
    )


    parser.add_argument(
        "--codebook_size",
        type=int,
        default=128,
        help="Dataloaders threads (default: %(default)s)",
    )


    parser.add_argument(
        "--num_images_val",
        type=int,
        default=400,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0018,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="/scratch/vqlic/models", help="Where to Save model"
    )

    parser.add_argument(
        "--name_folder", type=str, default="q6", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt



def main():
    #args = parse_args(argv)


    
    model_path = "/scratch/nvc/i_fram_net/cvpr2024_image.pth"
    ec_thread = False
    stream_part_i = 1
    device = "cuda"

    i_state_dict = get_state_dict(model_path)
    i_frame_net = DMCI(ec_thread=ec_thread, stream_part=stream_part_i, inplace=True)

    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)

    i_frame_net.update(force=True)


    print(i_frame_net)
    print("done")





