# OrdRec
This is our Tensorflow implementation for our TNNLS 2021 paper(under review) and some baselines:

>Bin Wu, Xiangnan He, Qi Zhang, Meng Wang, Yangdong Ye(2021). GCRec: Graph Augmented Capsule Network for Next-Item Recommendation(under review).

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.14.0
* numpy == 1.16.4
* scipy == 1.3.1
* pandas == 0.17

## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

NOTE: The cpp implementation is much faster than python.

## Examples to run OrdRec:
run [main.py](./main.py) in IDE or with command line:
```
python main.py
```

NOTE :  
 * the duration of training and testing depends on the running environment.
 * set model hyperparameters in .\conf\OrdRec.properties
 * set NeuRec parameters in .\NeuRec.properties       
 * the log files were saved at .\log\cellphones_tnnls\

## Dataset
We provide Amazon_Cell_Phones_and_Accessories(cellphones_tnnls) dataset.
  * .\dataset\cellphones_tnnls.rating
  * Each line is a user with her/his positive interactions with items: userID \ itemID \ ratings \ timestamp.
  * Each user has more than 10 associated actions.

## Baselines
The list of available models in TNNLS, along with their paper citations, are shown below:

| General Recommender | Paper                                                                                                         |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| BPRMF               | Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.                   |

| Sequential Recommender | Paper                                                                                                      |
|------------------------|------------------------------------------------------------------------------------------------------------|
| FPMC                   | Steffen Rendle et al., Factorizing Personalized Markov Chains for Next-Basket Recommendation, WWW 2010.    |
| HGN                    | Chen Ma et al., Hierarchical Gating Networks for Sequential Recommendation, KDD 2019.                      |
| SASRec                 | Wangcheng Kang et al., Self-Attentive Sequential Recommendation, ICDM 2018.     
| GCRec                  | Bin  Wu et al., GCRec: Graph Augmented Capsule Network for Next-Item Recommendation, IEEE TNNLS 2021.   

## Results
```
2021-09-10 15:29:51.111: Dataset name: cellphones
The number of users: 9534
The number of items: 53479
The number of ratings: 139141
Average actions of users: 14.59
Average actions of items: 2.60
The sparsity of the dataset: 99.972710%
2021-09-10 17:29:54.066: metrics:	Precision@10	Recall@10   	MAP@10      	NDCG@10     
   ...
2021-09-10 17:30:02.995   Epoch 1 : training time==[2.936s]
2021-09-10 17:30:06.031   Epoch 2 : training time==[3.036s]
    ...
2021-09-10 17:34:44.608   Epoch 96: training time==[3.025]
2021-09-10 17:34:46.986   testing time==[2.378s] results: 0.01014257  	0.03809093  	0.01660337  	0.02734699  
```
NOTE : the duration of training and testing depends on the running environment.



