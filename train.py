import torch
from torch.utils import data
import time
import cv2
import matplotlib.pyplot as plt

from Models.FRRN import FRRNet
from .utils.dataLoader import camvidLoader
from .utils.LossFunctions import *
from .utils.Score import *


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())

    # augmentation = aug

    data_path = '../input/dataset/CamVid'

    t_loader = camvidLoader(data_path, is_transform=True,
                            split='train',
                            )
    v_loader = camvidLoader(data_path, is_transform=True, split='val')

    n_classes = t_loader.n_classes

    trainloader = data.DataLoader(t_loader, batch_size=2, num_workers=16, shuffle=True)

    valloader = data.DataLoader(v_loader, batch_size=2, num_workers=16)

    running_metrics_val = runningScore(n_classes)
    # model = frrn(n_classes=12)

    model = FRRNet(in_channels=3, out_channels=12)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25, 30, 35, 38], gamma=0.1, verbose=True)
    # optimizer = optimizer_cls()

    loss_fn = cross_entropy2d
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    start_iter = 0

    best_iou = -100.0

    i = start_iter

    flag = True

    while i <= 40 and flag:
        i = i + 1
        print("Running Iteration Number: ", i)

        for index, (images, labels) in enumerate(trainloader):

            start_ts = time.time()
            model.train()
            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (index + 1) % 150 == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    index + 1,
                    len(trainloader),
                    loss.item(),
                    time_meter.avg / 2,
                )
                print(print_str)
                time_meter.reset()

            if (index + 1) % 150 == 0 or (i + 1) == 300:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in enumerate(valloader):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)
                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                    score, class_iou = running_metrics_val.get_scores()
                    for k, v in score.items():
                        print(k, v)
                    val_loss_meter.reset()
                    running_metrics_val.reset()
        scheduler.step()
    return model

def view_result(model,image,index):
    if index>11 or index <-1:
        print('out of range')
        return
    img = cv2.imread('../input/dataset/CamVid/test/Seq05VD_f04200.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # img = cv2.resize(img, (480, 360),interpolation = cv2.INTER_NEAREST)  # uint8 with RGB mode
    img = img[4:-4]
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    CAMVID_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]

    img = img.astype(float) / 255.0
    CAMVID_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]
    img -= CAMVID_MEAN
    img = img / CAMVID_STD
    # NHWC -> NCHW

    img = img.transpose(2, 0, 1)

    img = torch.from_numpy(img).float()

    img = img.unsqueeze(0)
    out = model(img.to(device)).squeeze(0)
    if index != -1:
        plt.imshow(out[1].detach().cpu().numpy())
    if index == -1:
        plt.imshow(camvidLoader.decode_segmap(out.max(0)[1].cpu().numpy()))
    return


if __name__ == '__main__':
    model = train()

