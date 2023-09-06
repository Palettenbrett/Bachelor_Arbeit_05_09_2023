from anomalib.models.efficientad import EfficientAD
import torch
from tqdm import tqdm
import time

if __name__ == "__main__":
    x = torch.rand(1,3,256,256).cuda()
    model = EfficientAD(teacher_out_channels=384, image_size=[256, 256], model_size="small").model.eval().cuda()

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

    # time_list = []
    # num = 1000
    # exclude_first = 10
    # for i in tqdm(range(num)):
    #     torch.cuda.synchronize()
    #     tic = time.time()
    #     model(x)
    #     torch.cuda.synchronize()
    #     time_list.append(time.time()-tic)
    # time_list = time_list[exclude_first:]
    # print("     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/(num-exclude_first))))