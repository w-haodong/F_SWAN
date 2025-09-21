# F-SWAN: Few-Shot Semi-supervised Wavelet Adaptation Network for Automated Cobb Angle Measurement

### Clinical deployment of foundation models for automated Cobb angle measurement faces significant challenges from domain shifts and limited data availability during the transition from conventional X-rays to low-dose EOS biplanar radiographs. In this work, we address the critical problem of adapting such large models in a demanding few-shot, semi-supervised scenario. To this end, we introduce F-SWAN, a novel adaptation framework that employs a truncated Segment Anything Model (SAM) encoder, fine-tuned efficiently via Low-Rank Adaptation (LoRA). The core of our method is a sophisticated semi-supervised domain adaptation (SSDA) strategy, realized through a dual-model (Explorer-Anchor) architecture. This strategy synergistically combines adversarial training, feature disentanglement, and a consistency regularization mechanism featuring a unique wavelet-based pseudo-label refinement process. F-SWAN demonstrates exceptional performance and data efficiency. On the public AASCE 2019 test set, it achieves a state-of-the-art Circular Mean Absolute Error (CMAE) of 2.33°, surpassing baselines that require manual cropping. Critically, when adapting to the target EOS domain, the framework achieves superior accuracy to fully supervised baselines while using only 10 labeled samples, supplemented by unlabeled data. Our work presents a robust and practical methodology for the few-shot adaptation of large vision models to specialized medical domains, substantially reducing the annotation burden for clinical deployment.

![flow_chart](https://github.com/user-attachments/assets/40d29c9d-6a51-4be1-a338-ed0513525b61)

#### Datasets: The dataset used in this paper is from AASCE 2019 (https://aasce19.grand-challenge.org/).

#### In this Github repository we provide the test code along with the pre-trained parameters file (shared via an online storage: model_10_200.pth, link: https://pan.baidu.com/s/13s4lfQI1F1JBxMucFjDXUg?pwd=tbej code: tbej). 



