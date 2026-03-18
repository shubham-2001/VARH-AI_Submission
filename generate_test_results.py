import os
import glob
import time
import torch
from models.team15_DSCF_Fused import DSCF_Fused
from utils import utils_image as util

def main():
    model_path = os.path.join('model_zoo', 'team15_DSCF_Fused.pth')
    model = DSCF_Fused(num_in_ch=3, num_out_ch=3, feature_channels=26, upscale=4)
    model.load_state_dict(torch.load(model_path), strict=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    data_dir = '/home/shubham/NTIRE_Project/competition_data/DIV2K_LSDIR_test_LR'
    save_dir = '/home/shubham/NTIRE_Project/VARH-AI_Results'
    os.makedirs(save_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
    
    # Warmup
    dummy = torch.zeros(1, 3, 256, 256).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    torch.cuda.synchronize()

    runtimes = []
    
    for p in img_paths:
        img_name = os.path.basename(p)
        out_name = img_name  # NTIRE 2026 rule: output file must match input exact name (e.g 0995x4.png -> 0995x4.png)
        
        img_lr = util.imread_uint(p, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, 1.0)
        img_lr = img_lr.to(device)
        
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            img_sr = model(img_lr)
        torch.cuda.synchronize()
        end = time.time()
        
        runtimes.append(end - start)
        img_sr = util.tensor2uint(img_sr, 1.0)
        util.imsave(img_sr, os.path.join(save_dir, out_name))
        print(f"Processed {out_name} in {(end - start)*1000:.2f} ms")

    avg_runtime = sum(runtimes) / len(runtimes)
    print(f"\nCompleted {len(runtimes)} images.")
    print(f"Average Runtime: {avg_runtime*1000:.3f} ms")
    
if __name__ == '__main__':
    main()
