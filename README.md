# HOCOBIS-AL
This repository contains the code and supplementary material for our paper "***H***ow To ***O***vercome **CO**nfirmation ***B***ias ***I***n ***S***emi-Supervised Image Classification By ***A***ctive ***L***earning" (ECML PKDD 2023)

We investigate the applicability of AL baselines in combination with SSL techniques. 
We identify 3 typical real-world challenges in image classification (Between-Class-Imbalance (BCI), Between-Class-Similarity (BCS) and Within-Class-Imbalance (WCI)) and demonstrate how SSL performance deteriorates when the labeled pool consists only of randomly chosen data due to confirmation bias. 
In addition, we demonstrate how simple active learning methods can overcome confirmation bias in SSL and significantly outperform results compared to passive selection in such challenging real-world scenarios.


| t-SNE Between-Class-Imbalance (BCI)                                                                                                                                                                          | t-SNE Between-Class-Similarity (BCS)                                                                                                                                                         | t-SNE Within-Class-Imbalance (WCI)                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ![](img/tsne_BCI.png)                                                                                                                                                                                        | ![](img/tsne_BCS.png)                                                                                                                                                                        | ![](img/tsne_WCI.png)                                                                                                                                                                                                                               |
| Random Sampling BCI                                                                                                                                                                                          | Random Sampling BCS                                                                                                                                                                          | Random Sampling WCI                                                                                                                                                                                                                                 |
| ![](img/BCI-MNIST_random.png)                                                                                                                                                                                | ![](img/BCS-MNIST_random.png)                                                                                                                                                                | ![](img/WCI-MNIST_random.png)                                                                                                                                                                                                                       |
| Confirmation Bias BCI                                                                                                                                                                                        | Confirmation Bias BCS                                                                                                                                                                        | Confirmation Bias WCI                                                                                                                                                                                                                               |
| Entropy of PL for original MNIST (orange) vs. entropy of PL for BCI-MNIST (blue). Confirming class imbalance by incorporating imbalanced pseudo-labels over and over. <br/>![](img/BCI_confirmationbias.png) | Amount of wrong PL for original MNIST (orange) vs BCS-MNIST (blue). Confirming wrong predictions by incorporating wrong pseudo-labels over and over. <br/> ![](img/BCS_confirmationbias.png) | Even though correctness of PL in WCI-MNIST (blue line) is better than in original MNIST (orange line), the final accuracy is worse (markers) since only the same, easy concepts are confirmed over and over.<br/> ![](img/WCI_confirmationbias.png) |
| Pseudo-Labeling: <br/> ![](img/BCI-MNIST_pseudolabel_al.png)                                                                                                                                                 | Pseudo-Labeling: <br/> ![](img/BCS-MNIST_pseudolabel_al.png)                                                                                                                                 | Pseudo-Labeling: <br/> ![](img/WCI-MNIST_pseudolabel_al.png)                                                                                                                                                                                        |
| Flexmatch: <br/> ![](img/BCI-MNIST_flexmatch_al.png)                                                                                                                                                         | Flexmatch: <br/> ![](img/BCS-MNIST_flexmatch_al.png)                                                                                                                                         | Flexmatch: <br/> ![](img/WCI-MNIST_flexmatch_al.png)                                                                                                                                                                                                |
| Fixmatch: <br/> ![](img/BCI-MNIST_fixmatch_al.png)                                                                                                                                                           | Fixmatch: <br/> ![](img/BCS-MNIST_fixmatch_al.png)                                                                                                                                           | Fixmatch: <br/> ![](img/WCI-MNIST_fixmatch_al.png)                                                                                                                                                                                                  |
# Getting started

## Install Requirements
```
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Start MLFlow for Tracking
```
(venv) mlflow server --tracking_uri=${TRACKING_URI}
```

## Run Experiments
Start the experiment using the script ```executables/main.py```:

```
(venv) PYTHONPATH=. python executables/main.py --tracking_uri=${TRACKING_URI}
```

