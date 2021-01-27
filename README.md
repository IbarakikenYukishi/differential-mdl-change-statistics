# Differential MDL Change Statistics for Hierarchical Detection Algorithm.

## 1. About
This repository contains the implementation code of differential MDL change statistics (D-MDL) [1].
An alert system for COVID-19 change sign detection using the differential MDL change statistics has been made open in public.
https://ibarakikenyukishi.github.io/d-mdl-html/index.html

## 2. Environment
- CPU: 2.7 GHz Intel Core i5
- OS: macOS High Sierra 10.13.6
- Memory: 8GB 1867 MHz DDR3
- python: 3.6.4. with Anaconda.

## 3. How to Run
### Data Preparation
1. Download the data from the link below:
https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide

2. Rename the file to `covid_data.csv` and put it in `./data/covid_data.csv`.

### RUN
- Run the jupyter file at `./jupyter/symptom/D-MDL_SCAW_Exponential.ipynb` for the exponential modeling
- Run the jupyter file at `./jupyter/symptom/D-MDL_SCAW_Gaussian.ipynb` for the Gaussian modeling
- Run the jupyter file at `./jupyter/synthetic/synthetic_{abrupt,gradual}_{mean,variance}.ipynb` for the synthetic datasets

## 4. Author & Mail address
- Ryo Yuki (ryo_yuki@mist.i.u-tokyo.ac.jp)
- Linchuan Xu (linchuan_xu@mist.i.u-tokyo.ac.jp)
- Shintaro Fukushima (shintaro_fukushima@mist.i.u-tokyo.ac.jp)

## 5. Requirements & License
### Requirements
- numpy==1.15.0
- scipy==1.3.1

## 6. Note
For the calculation of the code length in the exponential modeling, we used the conditional NML (CNML) code length of Gaussian distributions [2,3,4] for residual errors.

## 7. License
This code is licensed under MIT License.

## 8. Reference
1. Yamanishi, K., Xu, L., Yuki, R., Fukushima, S., & Lin, C. H. (2020). Change Sign Detection with Differential MDL Change Statistics and its Applications to COVID-19 Pandemic Analysis. arXiv preprint arXiv:2007.15179.
2. Grünwald, P. D., & Grunwald, A. (2007). The minimum description length principle. MIT press.
3. Miyaguchi, K. (2017). Normalized Maximum Likelihood with Luckiness for Multivariate Normal Distributions. arXiv preprint arXiv:1708.01861.
4. Kaneko, R., Miyaguchi, K., & Yamanishi, K. (2017, December). Detecting changes in streaming data with information-theoretic windowing. In 2017 IEEE International Conference on Big Data (Big Data) (pp. 646-655). IEEE.
