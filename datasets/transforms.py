from torchvision import transforms

from datasets.APR import APRecombination, HybridAugmentPlusSingle, HybridAugmentSingle


normalize = transforms.Compose([
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

def train_transforms(_transforms, ks=3, sigma= 0.5, prob = 0.5):
    transforms_list = []
    if _transforms == 'apr':
        print('APR-Single.', _transforms)
        transforms_list.extend([
            transforms.RandomApply([APRecombination()], p=1.0),
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif _transforms == 'ha':
        print('HybridAugment-Single.', _transforms)
        transforms_list.extend([
            transforms.RandomApply([HybridAugmentSingle(ks=ks, sigma= sigma, prob= prob)], p=1.0),
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif _transforms == 'ha_p':
        print('HybridAugment++ -Single.', _transforms)
        transforms_list.extend([
            transforms.RandomApply([HybridAugmentPlusSingle(ks=ks, sigma= sigma, prob= prob)], p=1.0),
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        print("No singles augmentation.")
        transforms_list.extend([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    return transforms_list


def test_transforms():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return test_transform