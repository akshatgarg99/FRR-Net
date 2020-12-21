def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    augmentation = aug

    data_path = '/content/FRR-Net/CamVid'

    t_loader = camvidLoader(data_path, is_transform=True,
                            split='train', img_size=(512, 1024),
                            augmentation=augmentation)
    v_loader = camvidLoader(data_path, split='val', img_size=(512, 1024))

    n_classes = t_loader.n_classes

    trainloader = data.DataLoader(t_loader, batch_size=2, num_workers=16, shuffle=True)

    valloader = data.dataloader(v_loader, batch_size=2, num_workers=16)

    running_metrics_val = runningScore(n_classes)

    model = FRRNet(in_channels=3, out_channels=12)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device()))

    optimizer_cls = torch.optim.Adam(model.parameters(), lr=1.0e-4)

    loss_fn = cross_entropy2d
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    start_iter = 0

    best_iou = -100.0

    i = start_iter

    flag = True

    while i <= 300 and flag:
        for (images, labels) in trainloader:
            i = i + 1
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

            if (i + 1) % 10 == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    300,
                    loss.item(),
                    time_meter.avg / 2,
                )
                print(print_str)
                time_meter.reset()

            if (i + 1) % 10 == 0 or (i + 1) == 300]:
                model.eval()
            with orch.no_grad():
                for
            i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            outputs = model(images_val)
            val_loss = loss_fn(input=outputs, target=labels_val)



