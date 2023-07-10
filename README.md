# Adaptor
### Global Adaptive Transformer for Cross-Subject EEG Classification [[Paper](https://ieeexplore.ieee.org/document/10149036)]
##### Core idea: cross attention for distribution alignment
This model is somewhat bad, trying the effect of attention on domain adaptation.

## Abstract
![Network Architecture](/Fig1.png)

- We propose a Global Adaptive Transformer, named GAT, for domain adaptation in EEG classification, where cross attention is used to align marginal distributions of source and target domains (subjects).
- Parallel convolution branches are used to capture temporal and spatial features from raw EEG signals.
- We design an adaptive center loss to align the conditional distribution of EEG features.


## Requirements:
- Python 3.10
- Pytorch 1.12


## Datasets
- [BCI_competition_IV2a](https://www.bbci.de/competition/iv/) - acc 76.58% (hold out)
- [BCI_competition_IV2b](https://www.bbci.de/competition/iv/) - acc 84.44% (hold out)
<!-- - [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html) - acc 95.30% (5-fold) -->


Hope this code can be useful. I would appreciate you citing us in your paper. 😊
```
@article{song2023global,
  title = {Global {{Adaptive Transformer}} for {{Cross-Subject Enhanced EEG Classification}}},
  author = {Song, Yonghao and Zheng, Qingqing and Wang, Qiong and Gao, Xiaorong and Heng, Pheng-Ann},
  year = {2023},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume = {31},
  pages = {2767--2777},
  issn = {1558-0210},
  doi = {10.1109/TNSRE.2023.3285309}
}

```

