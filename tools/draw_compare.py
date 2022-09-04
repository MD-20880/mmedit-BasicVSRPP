from sysconfig import get_path
import matplotlib.pyplot as plt 
from PIL import Image
import argparse

def draw(lq_img,
         hq_img,
         gt_img,
         out_path,
         size = (720,360),
         title = "TItle"
         ):
    
    fig_num = 2
    lq = Image.open(lq_img)
    hq = Image.open(hq_img)
    lq = lq.resize(size)
    hq =hq.resize(size)
    if gt_img is not None:
        gt = Image.open(gt_img)
        gt =gt.resize(size)
        fig_num = 3
    
    

    fig,axs = plt.subplots(1,fig_num,figsize=(15,7))
    axs[0].set_title("LQ")
    axs[1].set_title("HQ")
    axs[0].imshow(lq)
    axs[1].imshow(hq)
    if gt_img is not None:
        axs[2].imshow(gt)
        axs[2].set_title("GT")
    fig.suptitle(title)
    fig.savefig(out_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lq_path")
    parser.add_argument("hq_path")
    parser.add_argument("--gt_path")
    parser.add_argument('outpath',type=str,default=None)
    
    args = parser.parse_args()
    
    return args
    
def main():
    args = parse_args()
    draw(args.lq_path,args.hq_path,args.gt_path,args.outpath)

draw(
    r"D:\Dataset\dataset_REDS\Synthetic4x_LLNoNoise\test\000\00000000.png",
    r"Images/Demoinput/basicvsr_plusplus_c64n7_8x1_600k_reds4_FromScratch_LLNN/test/000_LLNNImages/0.png",
    None,
    r"D:\mmediting\Images\LLNNmodel_20000iter_LLNNImage",
    title="LLNNmodel_20000iter_LLNNImage"
)