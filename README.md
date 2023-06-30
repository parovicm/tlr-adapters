This is the code for training target language-ready (TLR) task adapters with `torch` and `adapter-transformers`. Please refer to our paper [Cross Lingual Transfer with Target Language-Ready Task Adapters](https://arxiv.org/abs/2306.02767) for background.

## Installation

First, install `python` >= 3.9 and `pytorch` >= 1.12.1, e.g. using `conda`:
```
conda create -n tlr-env python=3.9
conda activate tlr-env
conda install pytorch==1.12.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Then download and install `tlr-adapters`:
```
git clone https://github.com/parovicm/tlr-adapters
cd tlr-adapters
pip install -e .
```

## Training TLR adapters

The code for training TLR adapters is given by the [`TLR trainer`](src/tlr/trainer.py) which modifies the original `Trainer` from the `adapter-transformers`. 

Scripts for all tasks supported are provided in [`examples/`](examples).

For example, the script for training the `TASK-MULTI` variant (see the [paper](https://arxiv.org/abs/2306.02767) for the explanation) for the `AmericasNLI` dataset is given [here](examples/text-classification/train_nli_multi.sh).
Generally, unless you are using a single language adapter during task adapter training (this is the case with `MAD-X` and `TARGET` variants) you need to pass the file containing language adapters to be used.
For the `TASK_MULTI` variant of the `AmericasNLI` dataset language adapters file will have the following content:

```
en PATH_TO_EN_LANG_ADAPTER
aym PATH_TO_AYM_LANG_ADAPTER
bzd PATH_TO_BZD_LANG_ADAPTER
cni PATH_TO_CNI_LANG_ADAPTER
gn PATH_TO_GN_LANG_ADAPTER
hch PATH_TO_HCH_LANG_ADAPTER
nah PATH_TO_NAH_LANG_ADAPTER
oto PATH_TO_OTO_LANG_ADAPTER
quy PATH_TO_QUY_LANG_ADAPTER
tar PATH_TO_TAR_LANG_ADAPTER
shp PATH_TO_SHP_LANG_ADAPTER
```

## Citation
If you use this code, please cite the following paper:

Marinela Parović, Alan Ansell, Ivan Vulić, and Anna Korhonen. 2023. [Cross Lingual Transfer with Target Language-Ready Task Adapters](https://arxiv.org/abs/2306.02767). In *Findings of the 61st Annual Meeting of the Association for Computational Linguistics.
