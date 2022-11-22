
# Approximate k-NN on augmented CIFAR-10 

The repository builds approximate k-nearest neighbour classifiers from CIFAR-10 dataset. We use a trivial $\ell_2$ metric on pixel space, but augment the training data with different data augmentation techniques. The experiments were motivated by the paper [Generalization to translation shifts: a study in architectures and augmentations](https://arxiv.org/abs/2207.02349). The paper explores how much data augmentation can capture the inbuilt priors of convolutional networks in more general purpose architectures like vision transformers and MLP-mixers. It is shown that even on a small dataset like CIFAR-10, all architectures get competitive performance when tranined with sufficiently advanced augmentation techniques. 

What is the limit of data augmentation in incorporating image specific priors. Here, we studied how much data augmentation helps with the most trivial non-linear classifier---nearest neighbour classifier in the pixel space with simple $\ell_2$ metric. We play with different approximate k-NN methods from the [faiss](https://github.com/facebookresearch/faiss) library as exact NN search is very slow. We also sweep over different values of $k=1,2,\ldots, 50, 60, 70, \ldots, 100, 200,\ldots, 1000$ for our k-NN.  Our findings are summarized below. Overall, we find that this trivial k-NN although is better in the augmented space, it is far from competitive to neural networks!! 

| Index | no_aug | basic_aug | adv_aug | adv_aug. + mixup |
| --- | --- | --- | --- | --- |
| pq(32,8) | 0.4170 (k=8)| 0.4398 (k=47)| - | 0.4945 (k=90)|
| ivfpq(10,32,8) | 0.3929 (k=10) | 0.4248 (k=42) | 0.4762 (k=200) | 0.4914 (k=70) | 
| ivfpq(10,32,8)+PCA 10 | 0.4100 (k=10) | 0.4373 (k=24) | 0.4880 (k=200) | 0.5015 (k=200)|

----
### Contributors and Acknowledgements
The experiments were conveived by Suriya Gunasekar and Nati Srebro. Tal Wagner helped with the implementation. We thank the open source libraries [faiss](https://github.com/facebookresearch/faiss) and [timm](https://github.com/rwightman/pytorch-image-models), which this repository builds from.



### Warnings
- This implementation is not optimized wrt memory and efficiency for datasets larger than CIFAR-10 
- Check requirements.txt file for required packages

### Usage
- The code uses [faiss](https://github.com/facebookresearch/faiss) library for fast approximate nearest neighbor implementation, and [timm](https://github.com/rwightman/pytorch-image-models) for implementations of advanced augmentations and mixup. Please check documentations therein for additional information.
- Check commanline options and default options in `config.py`
    - `--basic-augmentation` uses horizontal flip and random crop with 4 pixel padding. For each training image,  -- all combinations are included in building the ann classifier
    - `--advanced-augmentation` uses basic augmentation, along with RandAugment, RandomErasing. For each training image `--epochs` number of random transforms from this list are included in the ann classifier. 
    - `--use-mixup` creates ANN model on the space of images obtained after mixup/cutmix preprocessing from batches of data
    - `--indexes` single or list of faiss indices to use for ann algorithm. Check faiss library documentation for descriptions of the different options.
- Optionally, edit default options to your preference
- Run `main.py` with appropriate commanline arguments, e.g.,
    - `python main.py  --indexes pq ivfpq` for no augmentation
