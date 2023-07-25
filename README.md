# HybridAugment++: Unified Frequency Spectra Perturbations for Model Robustness (ICCV'23)

This repository contains the PyTorch implementation of our paper ["HybridAugment++: Unified Frequency Perturbations for Model Robustness"](https://arxiv.org/abs/2307.11823) accepted to [ICCV 2023](https://iccv2023.thecvf.com/) .


## Paper Abstract
*Convolutional Neural Networks (CNN) are known to exhibit poor generalization performance under distribution shifts. Their generalization have been studied extensively, and one line of work approaches the problem from a frequency-centric perspective. These studies highlight the fact that humans and CNNs might focus on different frequency components of an image. First, inspired by these observations, we propose a simple yet effective data
augmentation method HybridAugment that reduces the reliance of CNNs on high-frequency components, and thus improves their robustness while keeping their clean accuracy high. 
Second, we propose HybridAugment++, which is a hierarchical augmentation method that attempts to unify various frequency-spectrum augmentations. HybridAugment++ builds on HybridAugment, 
and also reduces the reliance of CNNs on the amplitude component of images, and promotes phase information instead. This unification results in competitive to or better than state-of-the-art results
on clean accuracy (CIFAR-10/100 and ImageNet), corruption benchmarks (ImageNet-C, CIFAR-10-C and CIFAR-100-C), adversarial robustness on CIFAR-10 and out-of-distribution detection on various datasets. HybridAugment
and HybridAugment++ are implemented in a few lines of code, does not require extra data, ensemble models or additional networks.*


## Paper Highlights
:pushpin: We propose *HybridAugment* and *HybridAugment++*, two simple data augmentation methods which force models to emphasize low-frequency components, and low-frequency/phase components of training samples, respectively. Both augmentations come with single-image and paired variants, which can and does work better in tandem. Such augmentations lead to models that are robust against various distribution shifts, while keeping or even improving the accuracy on clean samples.

<p align="center">
    <img src=./assets/fig1.png width="800">
</p>

 **Fig. 1: An overview of our methods HybridAugment (HA) and HybridAugment++ (HA ++ ), and their single image (_S) and paired (_P) variants. HA_P combines the high-frequency (HF) and low-frequency (LF) contents of two randomly selected images, whereas HA_P ++ combines the HF of one image with the amplitude and LF-phase mixtures of two other images. Single image variants perform the same procedure, but based on different augmented versions of a single image.**


:pushpin: Our methods outperform existing state-of-the-art methods on various benchmarks, including the corruption benchmark ImageNet-C.  HybridAugment++ improves its results with more training data (i.e. DeepAugment) and other augmentation methods (i.e. AugMix), and can be used to tailor any need by changing the cut-off frequency (i.e. higher clean accuracy vs higher robustness). We also show that our method is not exclusive to CNNs, and also works quite well with transformers.


<p align="center">
    <img src=./assets/fig2.png width="800">
</p>

 **Fig. 2: Clean error and corruption robustness on ImageNet. Lower is better. The methods shown in the last four rows leverage extra data during training. † indicates training with a higher cut-off frequency.**

:pushpin: *HybridAugment* and *HybridAugment++* are easy to implement, do not require extra data, ensemble models or complicated augmentation regimes based on external networks.


## Installation
:pushpin: See ```environment.yml``` file for an exported conda environment. Note that there might be unnecessary dependencies there, so the download might take a while.

:pushpin: See ```requirements.txt``` file for pip dependencies. Note that there might be unnecessary dependencies there, so the download might take a while.


### Datasets
:pushpin: Both CIFAR and imagenet training scripts look for the datasets under  ```./data/``` folder, though this can be changed with the relevant flags.

:pushpin: Links for some of the datasets: [CIFAR-10-C](https://zenodo.org/record/2535967), [CIFAR-100-C](https://zenodo.org/record/3555552), [ImageNet-C](https://zenodo.org/record/2235448).


## Running the code
:pushpin: Run the following script to train on CIFAR-10/100.
```
python main.py --outf output_folder --single_mode ha_p --paired_mode ha_p  --model "resnet" --dataset "cifar10"
```
> See the input args for the other options. use --eval to evaluate the trained model. The training/evaluation will be logged under the output_folder.
> This script will evaluate on both CIFAR-10/100 and their corrupted versions.

> Use --ood_dataset flag to choose which OOD dataset you would like to test on. Put these OOD datasets under ```./data/``` folder for easy experimentation.


:pushpin: Run the following script to train on ImageNet.
```
python imagenet.py --arch "resnet50" --data path/to/imagenet  --multiprocessing-distributed --rank 0 --world_size 1 --single_mode ha_p --paired_mode ha_p 
```
> See the input args for the other options. use --evaluate to evaluate a trained model on ImageNet. 
> This script will only evaluate on ImageNet.

:pushpin: For a fair comparison with other methods, we use the evaluation script of AugMix (see [here](https://github.com/google-research/augmix/blob/master/imagenet.py)). After downloading that repo, you can evaluate the ImageNet-trained model on ImageNet-C as follows.
```
python imagenet.py  --evaluate --resume path/to/checkpoint  path/to/imagenet path/to/imagenet_c
```
> These arguments should be fine for evaluation, but refer to the relevant script for more options.

:pushpin: Run the following scripts (under ```./autoattacks/``` folder) to train on CIFAR10 with adversarial training.
```
python train_fgsm.py --lr-max 0.20 --prob-p 0.16 --prob-s 0.90 --epochs 90 --out-dir output_folder --single_mode ha_p --paired_mode ha_p --opt-level O0
```
> See the script for more options during training.

:pushpin: Run the following scripts (under ```./autoattacks/``` folder) to evaluate the adversarial robustness of trained models (works with models trained with ```./train_fgsm.py/```).
```
python eval.py --model  path/to/model.pth  --data_dir ../data/cifar10/ --log_path path/to/log.txt
```
> See the script for more options during evaluation.

## Pretrained Weights
:pushpin: We provide pretrained weights as well as the training/evaluation logs for most of our models.


:pushpin: HybridAugment++ (PS) models (CIFAR-10).

|         | [AllConv](https://drive.google.com/drive/folders/13oMLXhDvIetDeTCAZqNyV838ThFLuQV5?usp=drive_link) | [DenseNet](https://drive.google.com/drive/folders/15TvDJvno28NfZT7eO0p33eB90jh16Dfj?usp=drive_link) | [WideResNet](https://drive.google.com/drive/folders/1FQ3YnVxCGFG0nHXTXicEkN6nRmk8h3gj?usp=drive_link)   | [ResNext](https://drive.google.com/drive/folders/1OiR5LeDiGLFdFTNyq5p5RzHItCbfpK_Q?usp=drive_link)   | [ResNet18](https://drive.google.com/drive/folders/1J7jIvMyMFdmObnq2o0hEJTWFytNaGISz?usp=drive_link) |
| :-------- |:---------: |:---------:| :----:| :----:| :---: |
| mCE   | 10.7      | 9.5     | 8.3 | 7.9 | 8.2 |

:pushpin: HybridAugment++ (PS) models (CIFAR-100).


|         | [AllConv](https://drive.google.com/drive/folders/1C1y4N2a7-XkwyaxxHRgRg5D83MRB2eQu?usp=drive_link) | [DenseNet](https://drive.google.com/drive/folders/1DQDY7JCE8LsShvCqYn2rb18422qlfNO0?usp=drive_link) | [WideResNet](https://drive.google.com/drive/folders/11NvcipDtRsUOlsJS2K3q2gihKgj_DWyg?usp=drive_link)   | [ResNext](https://drive.google.com/drive/folders/1zC4GP_GbaIWK4mLyOTF-j_ExR4qM-sXe?usp=drive_link)   | [ResNet18](https://drive.google.com/drive/folders/1knFV_hbrnq6eSigEv8MMhVeq5XOtpCQP?usp=drive_link) |
| :-------- |:---------: |:---------:| :----:| :----:| :---: |
| mCE   | 34.4      | 33.4     | 31.2 | 28.8 | 29.9 |


:pushpin: Pretrained models on ImageNet (ResNet50).

|         | [HA++ (PS)](https://drive.google.com/file/d/1SpRU3oU3lZAuNDD-ncNkKN4Nbxnbfkoq/view?usp=drive_link) | [HA++ (PS) †](https://drive.google.com/file/d/1wonGxX4UEJwuu65jaRXRUvniZ2jI_LZR/view?usp=drive_link) | [HA++ (PS)  + DA](https://drive.google.com/file/d/1O56wJoBo8cf0G-wGpWGGDhpGLji8ojwU/view?usp=drive_link)   | [HA++ (PS)  + DA †](https://drive.google.com/file/d/1fGtSRnW8ly_rAF_C_YNznVNOosb7T6p-/view?usp=drive_link)    | [HA++ (PS)  + DA + AM †](https://drive.google.com/file/d/1lDZyfzNxuo2cK37S356CuRutLjyJpVJB/view?usp=drive_link)  |
| :-------- |:---------: |:---------:| :----:| :----:| :---: |
| mCE   | 67.3      | 65.8     | 58.9 | 58.1 | 56.1 |


:pushpin: Models trained with adversarial training + our methods on CIFAR-10 (See Table 4 our paper).

|         | [HA (S)](https://drive.google.com/drive/folders/1lmu4bxaQAoOwKvYaXfkG7zrlZCNHPvSW?usp=drive_link) | [HA++ (S)](https://drive.google.com/drive/folders/1EKcPtMrgXPA_K3MYlxU-f-awc14g7P9G?usp=drive_link) | [HA (P)](https://drive.google.com/drive/folders/13oAKs2hiTfDHDR7sS_DcZpQyidM2V7zz?usp=drive_link)   | [HA++ (P)](https://drive.google.com/drive/folders/1lKpO4VMstkLKzHg-3AEPGjVaO_j0816E?usp=drive_link)    | [HA (PS)](https://drive.google.com/drive/folders/1b-_ESTtHs4I9_ywv8ZmpZ1wISXNjWYIO?usp=drive_link)  | [HA++ (PS)](https://drive.google.com/drive/folders/1IpTz8JdFbArLu54v8vIDCQAN5pxOH3Nc?usp=drive_link)  |
| :-------- |:---------: |:---------:| :----:| :----:| :---: | :---: | 
| CA   |  86.5    | 85.0     | 85.5 | 85.4 | 85.0 | 82.8 | 
| RA   |  44.1      | 45.4     | 42.1 | 43.5 | 44.8 | 46.0  | 


:exclamation: PS indicates paired-single combined variant. † indicates training with a higher cut-off frequency. DA is [DeepAugment](https://github.com/hendrycks/imagenet-r/tree/master/DeepAugment), AM is [AugMix](https://github.com/google-research/augmix).



## Citation
:pushpin: If you find our code or paper useful in your research, please consider citing our paper.
```bibtex
@inproceedings{yucel2023hybridaugment,
  title={HybridAugment++: Unified Frequency Spectra Perturbations for Model Robustness},
  author={Yucel, Mehmet Kerim and Cinbis, Ramazan Gokberk and Duygulu, Pinar},
  booktitle = {International Conference on Computer Vision (ICCV)}
  year={2023},
}
```


## Acknowledgements
This code base has borrowed several implementations from this [link](https://github.com/iCGY96/APR)


