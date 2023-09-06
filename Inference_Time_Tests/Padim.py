from anomalib.models.padim import Padim
import torch
from tqdm import tqdm
import time

if __name__ == "__main__":
    x = torch.rand(1,3,256,256).cuda()
    #model = Padim(input_size=[256, 256], layers=["layer1", "layer2", "layer3"], backbone="resnet18", n_features=100).model.eval().cuda()

    model = Padim.load_from_checkpoint("C:/Users/PaulR/anomalib/results/padim_v6/padim/parkscheibe/run/weights/lightning/model.ckpt", 
                                        input_size=[256, 256], layers=["layer1", "layer2", "layer3"], backbone="resnet18", n_features=100).eval().cuda()

    preds = 1000
    leftout = 10
    for i in tqdm(range(preds)):
        if i == 9:
            start = time.time() 
        model(x)

    stop = time.time()  
    elap = stop-start
    Frames = (preds-leftout)/elap
    time_per_frame = 1/Frames
    print(Frames)
    print(time_per_frame)