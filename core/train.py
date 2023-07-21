import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import AverageMeter

import numpy as np
import random
from datasets.transforms import normalize
import torchvision.transforms as T



def hybrid_augment_p(x, use_cuda=True, prob= 0.6,  ks= 3, sigma= 0.5):
    p = random.uniform(0, 1)

    if p > prob:
        return x

    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    blurrer = T.GaussianBlur(kernel_size=ks, sigma=sigma)

    lfc = blurrer(x)

    hfc = x - lfc
    hfc_mix = hfc[index]

    lfc = mix_data(lfc)
    hybrid_ims = lfc + hfc_mix


    return hybrid_ims

# APR-Pair
def mix_data(x, use_cuda=True, prob=0.6):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    p = random.uniform(0, 1)

    if p > prob:
        return x

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.fftn(x, dim=(1,2,3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    #randomly permute elements
    fft_2 = torch.fft.fftn(x[index, :], dim=(1,2,3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    #take the permuted amplitudes and mix with original phases.
    fft_1 = abs_2*torch.exp((1j) * angle_1)

    mixed_x = torch.fft.ifftn(fft_1, dim=(1,2,3)).float()

    return mixed_x

def hybrid_augment(x, use_cuda=True, prob= 0.6,  ks= 3, sigma= 0.5):
    p = random.uniform(0, 1)

    if p > prob:
        return x

    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    blurrer = T.GaussianBlur(kernel_size=ks, sigma=sigma)

    lfc = blurrer(x)
    hfc = x - lfc
    hfc_mix = hfc[index]
    hybrid_ims = lfc + hfc_mix

    return hybrid_ims

def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            inputs, targets = data.cuda(), labels.cuda()

        if options["paired_mode"] == "ha":
            inputs_mix = hybrid_augment(inputs, ks = options["kernel_size"], sigma = options["sigma"],
                                                 prob= options["prob"])
            if epoch==0 and batch_idx==0:
                print("HybridAugment-Paired.")
        elif options["paired_mode"] == "apr":
            inputs_mix = mix_data(inputs)
            if epoch == 0 and batch_idx==0:
                print("APR-Paired")
        elif options['paired_mode'] == "ha_p":
            inputs_mix = hybrid_augment_p(inputs, ks = options["kernel_size"], sigma = options["sigma"],
                                                prob= options["prob"])
            if epoch==0 and batch_idx==0:
                print("HybridAugment++ - Paired.")
        elif options["paired_mode"] == "none":
            inputs_mix = inputs.clone()
            if epoch==0 and batch_idx==0:
                print("No paired augmentations.")
        else:
            raise NotImplementedError

        inputs_mix = Variable(inputs_mix)
        batch_size = inputs.size(0)
        inputs, inputs_mix = normalize(inputs), normalize(inputs_mix)
        inputs = torch.cat([inputs, inputs_mix], 0)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            _, y = net(inputs, True)
            
            if options["paired_mode"] == "none":
                loss = criterion(y[:batch_size], targets)
            else:
                loss = criterion(y[:batch_size], targets) + criterion(y[batch_size:], targets)

            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), targets.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg

    return loss_all
