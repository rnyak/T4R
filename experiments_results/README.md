#  EXPERIMENT RESULTS

Here, we report the experiment results for three research questions that are stated in the paper. Experiments were conducted for baseline algorithms and the transformer-based architectures.

## eCommerce REES46

### Full Result Table

|Algorithm | NDCG@20  | HR@20  |
| :---:   | :-: | :-: |
| VSKNN |  | 283 |
| VSTAN | 301 | 283 |
| GRU4Rec(full train) | 301 | 283 |
| GRU4Rec(sliding window) | 301 | 283 |
 

## G1 news                                              

### Full Result Table


## Adressa News         

### Full Result Table


## Yoochoose Ecommerce

### Full Result Table


## Hyperparameter Optimization

For each set of algorithm, configuration, and dataset, we optimize hyperparametersfor 100 trials towards maximizing NDCG@20 of the validation set.
