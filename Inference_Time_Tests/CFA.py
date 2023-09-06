from anomalib.models.cfa import Cfa
import torch
from tqdm import tqdm
import time

if __name__ == "__main__":

    x = torch.rand(1,3,256,256).cuda()
    
    model = Cfa.load_from_checkpoint("C:/Users/PaulR/anomalib/results/cfa/cfa/parkscheibe/run/weights/lightning/model.ckpt", 
                                        input_size=[256, 256], backbone="wide_resnet50_2", num_nearest_neighbors=3, num_hard_negative_features=3, radius = 1.0e-05 ).eval().cuda()

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