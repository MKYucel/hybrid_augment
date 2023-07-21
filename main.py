import os
import sys
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from datasets import CIFAR10D, CIFAR100D, SVHNOOD, LSUNOOD, ImageNetOOD
from utils.utils import AverageMeter, Logger, save_networks, load_networks
from core import train, test, test_robustness

parser = argparse.ArgumentParser("Training")

# dataset
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./results')

parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('--workers', default=8, type=int, help="number of data loading workers (default: 4)")

# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--stepsize', type=int, default=30)

# model
parser.add_argument('--model', type=str, default='wider_resnet_28_10')

# misc
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)

# parameters for generating adversarial examples
parser.add_argument('--epsilon', '-e', type=float, default=0.0157,
                    help='maximum perturbation of adversaries (4/255=0.0157)')
parser.add_argument('--alpha', '-a', type=float, default=0.00784,
                    help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', '-k', type=int, default=10,
                    help='maximum iteration when generating adversarial examples')
parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf',
                    help='the type of the perturbation (linf or l2)')

#hybrid-image params
parser.add_argument('--paired_mode', type=str, default='ha_p',
                    help='ha, apr, ha_p, none. Choose the paired augmentation.')
parser.add_argument("--kernel_size", type=int, default=3,
                    help="(Square) kernel size.")
parser.add_argument("--sigma", type=float, default= 0.5,
                    help="Default sigma cutoff.")
parser.add_argument("--prob", type=float, default=0.6,
                    help="Prob. of hybrid mixing.")
parser.add_argument("--kernel_size_single", type=int, default=3,
                    help="(Square) kernel size single.")
parser.add_argument("--sigma_single", type=float, default= 0.5,
                    help="Default sigma cutoff single.")
parser.add_argument("--prob_single", type=float, default=0.5,
                    help="Prob. of hybrid mixing single.")
parser.add_argument('--ood_dataset', type=str, default='cifar100',
                    help='Choose OOD dataset to be evaluated ["svhn", "cifar100", "lsun", "lsun_fix", "imagenet", "imagenet_fix"]'
                         '. DOES NOT WORK FOR CIFAR100 AS TRAINING SET')

parser.add_argument('--single_mode', type=str, default='ha_p', help='ha, apr, ha_p, none. Choose the singles augmentation')


args = parser.parse_args()
options = vars(args)

if not os.path.exists(options['outf']):
    os.makedirs(options['outf'])

if not os.path.exists(options['data']):
    os.makedirs(options['data'])


dump_name = "train_log.txt" if not options["eval"] else "eval_logs.txt"
if options["ood_dataset"] == "cifar100":
    dump_name = dump_name
elif options["ood_dataset"] == "svhn":
    dump_name = "svhn_" + dump_name
elif options["ood_dataset"] == "lsun":
    dump_name = "lsun_" + dump_name
elif options["ood_dataset"] == "lsun_fix":
    dump_name = "lsun_fix_" + dump_name
elif options["ood_dataset"] == "imagenet":
    dump_name = "imagenet_" + dump_name
elif options["ood_dataset"] == "imagenet_fix":
    dump_name = "imagenet_fix_" + dump_name

sys.stdout = Logger(osp.join(options['outf'], dump_name))

def main():
    print(options)
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    options.update({'use_gpu': use_gpu})

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    if 'cifar10' == options['dataset']:
        Data = CIFAR10D(dataroot=options['data'], batch_size=options['batch_size'], _transforms=options['single_mode'],
                        num_workers= options["workers"], _eval=options['eval'], ks_single= options["kernel_size_single"], sigma_single= options["sigma_single"],
                        prob_single = options["prob_single"])

        if options["ood_dataset"] == "cifar100":
            OODData = CIFAR100D(dataroot=options['data'], batch_size=options['batch_size'],
                                num_workers=options["workers"], _transforms=options['single_mode'])
            print("OOD dataset: CIFAR100")
        elif options["ood_dataset"] == "svhn":
            OODData = SVHNOOD(dataroot=options['data'], batch_size=options['batch_size'],
                            num_workers= options["workers"])
            print("OOD dataset: SVHN")
        elif options["ood_dataset"] == "lsun":
            OODData = LSUNOOD(dataroot=options['data'], batch_size=options['batch_size'],
                            num_workers= options["workers"])
            print("OOD dataset: LSUN")
        elif options["ood_dataset"] == "lsun_fix":
            OODData = LSUNOOD(dataroot=options['data'], batch_size=options['batch_size'],
                            num_workers= options["workers"], fixed_version= True)
            print("OOD dataset: LSUN-fix")
        elif options["ood_dataset"] == "imagenet":
            OODData = ImageNetOOD(dataroot=options['data'], batch_size=options['batch_size'],
                              num_workers=options["workers"])
            print("OOD dataset: ImageNet")
        elif options["ood_dataset"] == "imagenet_fix":
            OODData = ImageNetOOD(dataroot=options['data'], batch_size=options['batch_size'],
                              num_workers=options["workers"], fixed_version = True)
            print("OOD dataset: ImageNet-fix")
        else:
            raise NotImplementedError
    else:
        Data = CIFAR100D(dataroot=options['data'], batch_size=options['batch_size'], _transforms=options['single_mode'],
                         num_workers= options["workers"], _eval=options['eval'], ks_single= options["kernel_size_single"], sigma_single= options["sigma_single"],
                         prob_single = options["prob_single"])
        OODData = CIFAR10D(dataroot=options['data'], batch_size=options['batch_size'],
                           num_workers= options["workers"], _transforms=options['single_mode'])
    
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, OODData.test_loader
    num_classes = Data.num_classes

    if 'wide_resnet' in options['model']:
        print('wide_resnet')
        from model.wide_resnet import WideResNet
        net = WideResNet(40, num_classes, 2, 0.0)
    elif 'allconv' in options['model']:
        print('allconv')
        from model.allconv import AllConvNet
        net = AllConvNet(num_classes)
    elif 'densenet' in options['model']:
        print('densenet')
        from model.densenet import  densenet
        net = densenet(num_classes=num_classes)
    elif 'resnext' in options['model']:
        print('resnext29')
        from model.resnext import resnext29
        net = resnext29(num_classes)
    else:
        print('resnet18')
        from model.resnet import ResNet18
        net = ResNet18(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss().cuda()

    if use_gpu:
        net = nn.DataParallel(net, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
        criterion = criterion.cuda()

    file_name = '{}_{}_{}'.format(options['model'], options['dataset'], options['single_mode'])

    if options['eval']:
        net, criterion = load_networks(net, options['outf'], file_name, criterion=criterion)
        outloaders = Data.out_loaders
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        acc = results['ACC']
        res = dict()
        res['ACC'] = dict()
        acc_res = []
        for key in Data.out_keys:
            results = test_robustness(net, criterion, outloaders[key], epoch=0, label=key, **options)
            print('{} (%): {:.3f}\t'.format(key, results['ACC']))
            res['ACC'][key] = results['ACC']
            acc_res.append(results['ACC'])
        print('Mean ACC:', np.mean(acc_res))
        print('Mean Error:', 100-np.mean(acc_res))

        return

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]


    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.2, milestones=[60, 120, 160, 190])

    start_time = time.time()

    best_acc = 0.0
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch'] or epoch > 160:
            print("==> Test")
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)

            if best_acc < results['ACC']:
                best_acc = results['ACC']
                print("Best Acc (%): {:.3f}\t".format(best_acc))
                save_networks(net, options['outf'], "best_model", criterion=None)
                print("Saved best val model at epoch:", epoch+1)

            save_networks(net, options['outf'], file_name, criterion=criterion)

        scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

if __name__ == '__main__':
    main()

