from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as F
import torch.utils.data as Data

def train(train_loader, net, criterion, optimizer):
    net.train()
    train_loss, correct, total = 0., 0., 0.
    n = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % (n//20) == 0:
            print('%d/20 passed | Train Loss: %.3f, Acc: %.3f%%'.%(
                           batch_idx//(n//20), 
                           train_loss/(batch_idx+1),
                           100.*correct/total)
                    )
    print('Epoch finished!')
    return

def val(val_loader, net, criterion):
    net.eval()
    val_loss, correct, total = 0., 0., 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test set loss: %.3f | Acc: %.3f'%(val_loss/(batch_idx+1), 100.*correct/total))
    return 100.*correct/total


class loader(Data.Dataset):
    def __init__(self, list_file, size=(160,128), crop=112,
                 num_frames=32, dilate=1, test=False, 
                 mean = (0.455, 0.430, 0.396), 
                 std = (0.244, 0.237, 0.241)):
        self.list_file = open(list_file).readlines()
        self.size = size
        self.crop = crop
        self.test = test
        self.mean = mean
        self.std = std
        self.num_frames = num_frames
        self.dilate = dilate

    def get_index(img_num, num_frames=self.num_frames,
                   dilate=self.dilate):
        if img_num>=(num_frames+1)*dilate:
            start = int(torch.randint(img_num-(num_frames+1)*dilate+1))
            indexes = range(start,start+num_frames*dilate,dilate)
        elif img_num>=num_frames+1:
            dilate = img_num//(num_frames+1)
            start = int(torch.randint(img_num-(num_frames+1)*dilate+1))
            indexes = range(start,start+num_frames*dilate,dilate)
        else:
            indexes = torch.randint(img_num,size=(num_frames))
            indexes.sort()
        return [i+1 for i in indexes]

    def train_trans(img, size, box, 
                    brightness, contrast, saturation, hue, 
                    mean, std):
        img = img.resize(size)
        img = img.crop(box)
        img = F.adjust_brightness(img，brightness)
        img = F.adjust_contrast(img, contrast)
        img = F.adjust_saturation(img, saturation)
        img = F.adjust_hue(img, hue)
        img = F.to_tensor(img)
        img = F.normalize(tensor, mean, std)
        return img

    def test_trans(img, size, mean, std):
        img = img.resize(size)
        img = F.to_tensor(img)
        img = F.normalize(tensor, mean, std)
        return img

    def __getitem__(self, index):
        video_id, img_num, label = self.list_file[index].split('\t')
        if not self.test:
            x, y = size[0] - crop, size[1] - crop
            x = int(torch.randint(x,size=(1,)))
            y = int(torch.randint(y,size=(1,)))
            box = (x,y,x+self.crop,y+self.crop)
            bri, con, sat, hue = np.array(torch.rand(size=(4,)))/2. + 0.75
            hue = (hue-1)/2.
        sample_index = self.get_index(img_num)
        video = None
        for img_item in sample_index:
            path = '%s/%d/%05d.jpg'%(self.path, video_id, img_item+1)
            img = Image.open(path)
            if self.test:
                img = test_trans(img, self.size, self.mean, self.std)
            else:
                img = train_trans(img, self.size, box,
                                  bri, con, sat, hue,
                                  self.mean, self.std)
            video = img if video==None else torch.stack([video, img])
        return video, int(label)

    def __len__(self):
        return len(self.list_file)