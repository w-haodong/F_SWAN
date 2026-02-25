# Few-shot Semi-supervised Wavelet Adaptation for Automated Cobb Angle Measurement in Full-body EOS Radiographs

## Abstract 
Low-dose biplanar radiographs acquired by the EOS imaging system (EOS) are increasingly adopted for scoliosis follow-up because they provide weight-bearing full-body imaging, enabling clinicians to assess spinal curvature together with global alignment cues from the head, shoulders, pelvis, and lower limbs. In this practical setting, automated Cobb angle measurement must operate directly on uncropped full-body scans, yet robust model development is hindered by the scarcity of expert-annotated EOS data and the domain and field-of-view gap to public narrow-view radiographs. We propose F-SWAN (Few-shot Semi-supervised Wavelet Adaptation Network), which transfers a contour-based Cobb measurement model from the Accurate Automated Spinal Curvature Estimation 2019 benchmark (AASCE 2019) to EOS under limited target labels and further exploits unlabeled EOS scans to improve full-image robustness. F-SWAN represents each vertebra using a compact multi-scale wavelet descriptor and optimizes a geometry-aware reconstruction objective for stable contour recovery, while an Explorer--Anchor pseudo supervision scheme regularizes pseudo targets in the wavelet domain to encourage anatomically plausible geometry under cluttered full-body backgrounds. Experiments on AASCE 2019 and a private clinical EOS cohort demonstrate improved full-body Cobb angle estimation without manual cropping; notably, with only 10 labeled EOS images, F-SWAN achieves strong accuracy and approaches the fully labeled setting. 

<img width="1174" height="538" alt="image" src="https://github.com/user-attachments/assets/01e6fbfb-3016-4dec-b81c-6ff50e6fcf1e" />
<img width="1180" height="499" alt="image" src="https://github.com/user-attachments/assets/6c016a0a-6e31-49c9-ab56-14c525f892e1" />


## Acknowledgements
We thank Meta AI for making the source code of Dino V3 publicly available. ([Dino: https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3))

## Datasets
The dataset used in this paper is from AASCE 2019 (https://aasce19.grand-challenge.org/).

## Pre-trained parameters
In this Github repository we provide the test code along with the pre-trained parameters file (shared via an online storage: model_latest.pth, link: https://pan.baidu.com/s/1O-60eWuQZnPZgjqWDl-f2w?pwd=6a5k  code: 6a5k). 



