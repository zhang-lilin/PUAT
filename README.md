### Provable Unrestricted Adversarial Training without Compromise with Generalizability

Authors: Lilin Zhang, [Ning Yang](https://yneversky.github.io), Yanchao Sun, Philip S. Yu

------

This is official PyTorch implementation code for the paper of [Provable Unrestricted Adversarial Training without Compromise with Generalizability](https://ieeexplore.ieee.org/document/10530438) accepted by Transactions on Pattern Analysis and Machine Intelligence. 

In this paper, we propose a unique viewpoint to understand Unrestricted Adversarial Example (UAE) and a novel adversarial training method called Provable Unrestricted Adversarial Training (PUAT), which utilizes semi-supervised data to achieve comprehensive adversarial robustness with less reduce of standard generalization. The theoretical analysis and experiments demonstrate the superiority of PUAT. Refer to our paper for more details.

### Datasets

- [CIFAR10\CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [SVHN](http://ufldl.stanford.edu/housenumbers/)
- [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet/overview)
- [ImageNet32](http://image-net.org/download-images)

### Baseline Models

- [WideResNet-28-10](https://arxiv.org/abs/1605.07146) (WRN-28-10) with the [swish](https://arxiv.org/pdf/1606.08415) activation function

### Example usage

- Training a WRN-28-10 by PUAT on CIFAR10

```
python runner.py
```

### Citation

```
@article{
    zhang2024provable,
    author = {Zhang, Lilin and Yang, Ning and Sun, Yanchao and Philip, S Yu},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    publisher = {IEEE},
    title = {Provable Unrestricted Adversarial Training without Compromise with Generalizability},
    year = {2024}
}
```