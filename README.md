# Adaptor
### Global Adaptive Transformer for Cross-Subject EEG Classification [[Paper](https://ieeexplore.ieee.org/document/9991178)]
##### Core idea: cross attention for distribution alignment

## Abstract
![Network Architecture](/Fig1.png)

- We propose a Global Adaptive Transformer, named GAT, for domain adaptation in EEG classification, where cross attention is used to align marginal distributions of source and target domains (subjects).
- Parallel convolution is used to capture temporal and spatial features while preserving the structure information of EEG signals.
- We design an adaptive center loss to align the conditional distribution of EEG features.
- 

## Requirmenets:
- Python 3.10
- Pytorch 1.12


## Datasets
- [BCI_competition_IV2a](https://www.bbci.de/competition/iv/) - acc 76.58% (hold out)
- [BCI_competition_IV2b](https://www.bbci.de/competition/iv/) - acc 84.44% (hold out)
<!-- - [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html) - acc 95.30% (5-fold) -->


<!-- ## Citation
Hope this code can be useful. I would be very appreciate if you cite us in your paper. 😊
```
@article{song2023eeg,
  title = {{{EEG Conformer}}: {{Convolutional Transformer}} for {{EEG Decoding}} and {{Visualization}}},
  shorttitle = {{{EEG Conformer}}},
  author = {Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  year = {2023},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume = {31},
  pages = {710--719},
  issn = {1558-0210},
  doi = {10.1109/TNSRE.2022.3230250}
}
```  -->

