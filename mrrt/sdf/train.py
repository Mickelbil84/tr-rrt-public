import tqdm
import numpy as np

import torch

DELTA = 0.1

def train(data, model, device, criterion, optimizer, epoch, train_losses, vis=None):
    total_loss = 0
    cnt = 0
    train_enum = tqdm.tqdm(enumerate(data), desc='Train epoch %d' % epoch)

    for i, data in train_enum:
        xyz = data['xyz'].to(device)
        sdf = data['sdf'].to(device)
        sdf = torch.clamp(sdf, -DELTA, DELTA)

        batch_size = xyz.shape[0]

        model.zero_grad()
        output = torch.clamp(model(xyz), -DELTA, DELTA)
        loss = criterion(output, sdf)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        cnt += 1
        if i % 100 == 0:
            train_enum.set_description('Train (loss %.8f) epoch %d' % (total_loss / (i+1), epoch))
            train_losses.append(total_loss / (i+1))
            if vis is not None:
                vis.line(Y=np.asarray(train_losses), X=torch.arange(1, 1+len(train_losses)),
                    opts={'title': 'Loss'}, name='loss', win='loss')