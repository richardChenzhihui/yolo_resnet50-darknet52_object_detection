import os
import tqdm
import numpy as np
import argparse

import torch
import torchvision
from torchvision import transforms

from data.dataset import Dataset
from model.hkudetector import resnet50
from utils.loss import yololoss

# parse args
parser = argparse.ArgumentParser()
# S need to be 20 if input image size is 640
parser.add_argument('--yolo_S', default=14, type=int, help='YOLO grid num')
parser.add_argument('--yolo_B', default=2, type=int, help='YOLO box num')
parser.add_argument('--yolo_C', default=5, type=int, help='detection class num')

parser.add_argument('--num_epochs', default=40, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=3, type=int, help='batch size')
parser.add_argument('--learning_rate', default=2e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay')
parser.add_argument('--seed', default=666, type=int, help='random seed')
parser.add_argument('--dataset_root', default='./data/ass1_dataset', type=str, help='dataset root')
parser.add_argument('--output_dir', default='checkpoints', type=str, help='output directory')

parser.add_argument('--l_coord', default=5., type=float, help='hyper parameter for localization loss')
parser.add_argument('--l_noobj', default=0.5, type=float, help='hyper parameter for no object loss')
parser.add_argument('--continue_train', default=0, type=int, help='whether to continue train with best checkpoint model')
parser.add_argument('--load_path', default="./checkpoints/hku_mmdetector_best.pth", type=str, help='load existing model')
parser.add_argument('--img_size', default=448, type=int, help='Training img size')

args = parser.parse_args()


def load_pretrained(net):
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet_state_dict = resnet.state_dict()

    net_dict = net.state_dict()
    for k in resnet_state_dict.keys():
        if k in net_dict.keys() and not k.startswith('fc'):
            net_dict[k] = resnet_state_dict[k]
    net.load_state_dict(net_dict)


####################################################################
# Environment Setting
# We suggest using only one GPU, or you should change the codes about model saving and loading

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

# Other settings
args.load_pretrain = True
print(args)

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
####################################################################
criterion = yololoss(args, l_coord=args.l_coord, l_noobj=args.l_noobj)

hku_mmdetector = resnet50(args=args)
if args.load_pretrain:
    load_pretrained(hku_mmdetector)


if args.continue_train:
    model_path = args.load_path
    print('Continue training by loading%s' % (model_path))
    hku_mmdetector.load_state_dict(torch.load(model_path)['state_dict'])
hku_mmdetector = hku_mmdetector.to(device)



####################################################################
# Multiple GPUs if needed
# if torch.cuda.device_count() > 1:
#     torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
#     hku_mmdetector = torch.nn.parallel.DistributedDataParallel(hku_mmdetector)
if torch.cuda.device_count() > 1:
    hku_mmdetector = torch.nn.DataParallel(hku_mmdetector)

hku_mmdetector.train()

# initialize optimizer
optimizer = torch.optim.AdamW(hku_mmdetector.parameters(), betas=(0.9, 0.999), lr=args.learning_rate,
                              weight_decay=args.weight_decay)

# initialize dataset
train_dataset = Dataset(args, split='train', transform=[transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

###################################################################
# TODO: Please fill the codes below to initialize the validation dataset
##################################################################
val_dataset = Dataset(args, split='val', transform=[transforms.ToTensor()])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
##################################################################

print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
print(f'BATCH SIZE: {args.batch_size}')

train_dict = dict(iter=[], loss=[])
best_val_loss = np.inf

for epoch in range(args.num_epochs):
    hku_mmdetector.train()

    # training
    total_loss = 0
    print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
    progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

    # if epoch == 10:
    #     optimizer = torch.optim.AdamW(hku_mmdetector.parameters(), betas=(0.9, 0.999), lr=args.learning_rate*0.5)
    # if epoch == 20:
    #     optimizer = torch.optim.AdamW(hku_mmdetector.parameters(), betas=(0.9, 0.999), lr=args.learning_rate*0.25)
    # if epoch == 30:
    #     optimizer = torch.optim.AdamW(hku_mmdetector.parameters(), betas=(0.9, 0.999), lr=args.learning_rate*0.1)
    # if epoch == 30:
    #     learning_rate=0.0001
    # if epoch == 40:
    #     learning_rate=0.00001
    # # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = learning_rate

        
    for i, (images, target) in progress_bar:
        images = images.to(device)
        target = target.to(device)

        pred = hku_mmdetector(images)
        loss, loss_l = criterion(pred, target)
        if i == 100:
            print(loss_l)


        # print('target')
        # print(target.size())
        # print('pred')
        # print(pred.size())
        # print('loss')
        # print(loss)
        total_loss += loss.data

        ###################################################################
        # TODO: Please fill the codes here to complete the gradient backward
        ##################################################################
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        #torch.cuda.empty_cache()

        ##################################################################

        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, args.num_epochs), total_loss / (i + 1), mem)
        progress_bar.set_description(s)
    torch.cuda.empty_cache()
    # validation
    validation_loss = 0.0
    hku_mmdetector.eval()
    progress_bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (images, target) in progress_bar:
        images = images.to(device)
        target = target.to(device)

        prediction = hku_mmdetector(images)
        loss, loss_l = criterion(prediction, target)
        validation_loss += loss.data
    validation_loss /= len(val_loader)
    print("validation loss:", validation_loss.item())

    if best_val_loss > validation_loss:
        best_val_loss = validation_loss



        save = {'state_dict': hku_mmdetector.state_dict()}
        # for multiple gpu
        #save = {'state_dict': hku_mmdetector.module.state_dict()}

        torch.save(save, os.path.join(output_dir, 'hku_mmdetector_best.pth'))

    save = {'state_dict': hku_mmdetector.state_dict()}
    torch.save(save, os.path.join(output_dir, 'hku_mmdetector_epoch_'+str(epoch+1)+'.pth'))

    torch.cuda.empty_cache()
    