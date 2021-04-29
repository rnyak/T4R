# 4. Methodology

This section describes the training and evaluation protocols, including metrics and the hyperparameter tuning processused for the experiments. We also describe the datasets and their preprocessing.

##  4.1.1. Incremental Training

Online services receive a continuous stream of user interactions which makes the availabledataset larger every day. This scenario increases the time and computational resources required to train models andbrings engineering challenges for large-scale recommender systems.

In this work, like in [8,15,54], our experiments are performed with incremental retraining of the algorithms. Foreach algorithm, a sliding window with a single time unit (e.g. day or hour) is provided in temporal order and only once,to train the algorithms incrementally. For a nearest neighbor algorithm like V-SkNN, STAN or STAN (described inSection 4.1.6), this means that the algorithm needs to keep in memory a sample of sessions from past time windows.For neural networks, this means fine-tuning the parameters of model already trained with past data.

For the baseline that do not support incremental training / retraining – GRU4Rec – we used two training approachesthat trains an individual model for each evaluation time window 𝑇𝑖, retraining the model from scratch each time.
- Full Training (FT) - Trains a model on all available time windows prior to the evaluation time window i.e., from 𝑇1 to 𝑇𝑖−1.
- Sliding Window Training (SWT) - Trains a model with the last𝑤time windows, from 𝑇𝑖−𝑤 to 𝑇𝑖−1.

## 4.1.2. Incremental Evaluation

In an effort towards making our evaluations as realistic as possible, we do not use thecommon evaluation approach of random train-test splits and cross-validation. Instead the experiments are performedusing incremental evaluation, as done in [8,15,54]. This training and evaluation method allows us to emulate a commonproduction environment scenario where the recommendation algorithms are continuously trained and deployed once aday or even once an hour to serve recommendations for the next time period.

For our experiments sessions 𝑠∈𝑇 are split into time windows𝑇, with a length of one day for ecommerce datasetsand one hour for news datasets. Evaluation is performed for each next time window 𝑇𝑖+1 with𝑖∈ [1, . . .,𝑛−1], using sessions from past time windows for training[𝑇1, . . .,𝑇𝑖]. The sessions of each time window are split 50:50 between the validation set and test set. The time window validation sets are used for hyperparameter tuning and test sets for reporting metrics. The final reported metrics are the average of five independent runs with different random seeds,using the best configuration found in the hyperparameter optimization process (described next). For each run, the metrics are the averages of all time windows, i.e.,Average over Time (AoT), to benefit algorithms that are consistentproviding accurate recommendations over time.

## 4.1.3  Hyperparameter optimization

For each set of experiment group, which is composed by an algorithm, trainingapproach and dataset, we perform bayesian hyperparameter optimization for 100 trials – running five of them in parallel– towards maximizing NDCG@20 of the validation set. To reduce the possibility of overfitting over specific days, thehyperparameter tuning process is performed only in the first 50% of the days and the reported metrics are computed onthe test set for all days available in the datasets.

<center> Table 1. Dataset Statistics </center>

<center>

Dataset | Days | items (K) | sessions (M) | interactions (M) | sessions length (avg) | Gini index |
--- | --- | --- | --- |--- |--- |--- 
REES46 eCommerce| 31 | 156,516 | 3,268,268 | .. | .. | .. | 
YOOCHOOSE eCommerce | 182 | 50,549 | .. | .. | .. | .. | 
G1 news | 16 | 46,027 | .. | .. | .. | .. | 
ADRESSA news |  16 | 13,820 | 982,210 | .. | .. | .. | 

</center>


## 4.1.4 Metrics

We evaluate the algorithms for their ability to predict the last interacted item in a session. As sessionslengths range between 2 and 20 after preprocessing, this task is equivalent to next-click prediction, in the sense thatrecommendations for all positions of the session sequences will be evaluated. The following information retrievalmetrics are used to compute the Top-20 accuracy of recommendation lists containing all items: Normalized DiscountedCumulative Gain (NDCG@20)and Hit Rate(HR@20), which is equivalent toRecall@nwhen there is only one relevantitem in the recommendation list. The NDCG accounts for the rank of the relevant item in the recommendation list,whereas theHRjust verifies whether the relevant item is among the top-n items

## 4.1.5  Datasets and Pre-processing

We have selected for experiments the e-commerce and news domains, wheresession-based recommendation is very suitable. In the news domain, most users browse anonymously and usually onlytheir last interactions available. In the e-commerce domain, besides the user cold-start problem, user sessions tend to betargeted to a specific purchase need, so interactions from the current session provide more useful information than pastinteractions for the user context

We have selected two datasets for each of the domains, described as follows.

- REES46 eCommerce1- This dataset contains seven months of user session from a multi-category online store ,including events like views, add-to-card and purchase events. As this is a large dataset, we use in our experimentsonly events from the month of Oct. 2019.

- YOOCHOOSE eCommerce2- This dataset was released for the RecSys Challenge 2015 and is composed by clicksand buying events from user sessions on e-commerce. We use only the use interactions table (not the purchasesone), as they form the large majority of data and we are interested in next-click prediction.

- G1 news3- This dataset [8,15,44] was shared by globo.com, the most popular media company in Brazil. Itcontains sampled user sessions with page views of the G1 news portal during a period of 16 days . It also providesmetadata and a vectorial representation of the textual content of the news articles interacted during that period.

- ADRESSA news4- This news dataset [17] comes from collaboration of the NTNU and Adressavisen from Norway. It includes both page views and the textual content and metadata of news articles. We use only the first 16 daysof this dataset, so that its is comparable with the G1 news dataset.


Table 1 shows the statistics of these preprocessed datasets. It can be seen that the e-commerce datasets are largerthan the news datasets in all statistics. In special, the REES46 dataset have more sessions available per day and havelonger avg. session length. The statistics of the G1 and Adressa news dataset are very similar in general. Finally, theGini index of the items frequency distribution shows that the news dataset are more long-tailed, showing a popularitybias, with interactions more concentrated in a smaller set of very popular items.

### Preprocessing

All datasets provide session ids, except the ADRESSA dataset, for which we artificially split sessionsto have a maximum idle time of 30 minutes between the user interactions. We ignore sessions with length lower than 2,and truncate sessions up to the maximum of 20 interactions.

Repeated user interactions on the same items within sessions are removed from the news datasets, as they do notprovide information gain. Therefore, for the e-commerce domain, repeated interactions are common when users arecomparing products and recommending items already interacted can be helpful from a user’s perspective e.g., asreminders [25,29,40,50]. Thus for the e-commerce datasets we remove consecutive interactions in the same items, butallow them to be interleaved, e.g. the sequence of interacted items 𝑎𝑎𝑎𝑏𝑐𝑐𝑎 becomes 𝑎𝑏𝑐𝑎.

The sessions are divided in time windows, according to the unit: one day for e-commerce datasets and one hourfor the news datasets. The reason for different time units is that in the news domain the interactions are very biasedtoward recent items. For example, inG1 news, 50% and 90% of interactions are performed on articles published withinup to 11 hours and 20 hours, respectively. So, training those types of algorithms on a daily basis would not be effectivefor recommending fresh articles, as they would not be included in the train set.

### Feature Engineering
We explore the usage of side features by Transformers architectures (RQ3). So, we makeavailable some features for each dataset with simple feature engineering, as described in Table 2.

INSERT TABLE 2

## 4.1.6  Baseline Algorithms

An important aspect of any research paper are the baselines to which it is compared. Manyneural session-based recommendation papers only compare their results to other neural baselines. In order to validatethat Transformer architectures are effective for the task of session-based recommendation we have chosen to includethe highest performing algorithms from amongst both neural and non-neural baselines, selecting the top 3 of each fromthe evaluation done in [].One of the selected baseline algorithms is theVector Multiplication Session-Based k-NN (V-SkNN)algorithm [36].Although simple, V-SkNN has shown state-of-the-art performance compared to neural models like GRU4Rec, as reportedin [37] [36]. We use the V-SkNN algorithm implementation made available in [37]. To make V-SkNN support incrementaltraining, we updated the implementation to compute theIDFstatistics incrementally.We also used as a baseline theGated Recurrent Unit (GRU)[5], replacing the Transformer block from our proposedneural architecture (Figure 3) by this RNN. Such a setting allowed us to isolate the potential improvements obtained bythe usage of Transformers

One of the selected baseline algorithms is the Vector Multiplication Session-Based k-NN (V-SkNN)algorithm [36]. Although simple, V-SkNN has shown state-of-the-art performance compared to neural models like GRU4Rec, as reportedin [37] [36]. We use the V-SkNN algorithm implementation made available in [37]. To make V-SkNN support incrementaltraining, we updated the implementation to compute theIDFstatistics incrementally.We also used as a baseline theGated Recurrent Unit (GRU)[5], replacing the Transformer block from our proposedneural architecture (Figure 3) by this RNN. Such a setting allowed us to isolate the potential improvements obtained bythe usage of Transformers.

We also used as a baseline theGated Recurrent Unit (GRU)[5], replacing the Transformer block from our proposedneural architecture (Figure 3) by this RNN. Such a setting allowed us to isolate the potential improvements obtained bythe usage of Transformers.
