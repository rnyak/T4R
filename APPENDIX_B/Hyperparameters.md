#  Hyper-parameter Tuning

Here, we report the hyperparameter search space that is set and explored for each experiment. 

## Source code and Datasets

The full source code of the frameworks can be found here: 

https://github.com/rnyak/T4R/tree/main/codes

The datasets used in the experiments can be downloaded here: <br>

- https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store <br>
- https://2015.recsyschallenge.com/challenge.html <br>
- https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom <br>
- http://reclab.idi.ntnu.no/dataset/


<!DOCTYPE html>
<html>
<body>
<br>
<h3>Table 1. Transformers (Additional hyperparameters when using side information)</h3>
<font size="2" face="Arial" >
<table class="hp-table">
<thead><tr class="table-firstrow"><th>Experiment Group </th><th>Type </th><th>Name</th><th>Values</th><th>Distribution</th></tr></thead><tbody>
 <tr><td rowspan=3>Common hyperparameters</td><td>fixed</td><td>layer_norm_all_features</td><td>FALSE</td><td><center>-</center></td></tr>
 <tr><td>fixed</td><td>layer_norm_featurewise</td><td>TRUE</td><td><center>-</center></td></tr>
 <tr><td>hypertuning</td><td>embedding_dim_from_cardinality_multiplier</td><td>[1.0, 10.0]</td><td>discrete_uniform (step 1.0)</td></tr>
 <tr><td>Concatenation merge - Numericals features as scalars</td><td>fixed</td><td>input_features_aggregation</td><td>concat</td><td><center>-</center></td></tr>
 <tr><td rowspan=3>Concatenation merge - Numerical features - Soft One-Hot Encoding</td><td>fixed</td><td>input_features_aggregation</td><td>concat</td><td><center>-</center></td></tr>
 <tr><td>hypertuning</td><td>numeric_features_project_to_embedding_dim</td><td>[5, 55]</td><td>discrete_uniform (step 10)</td></tr>
 <tr><td>hypertuning</td><td>numeric_features_soft_one_hot_encoding_num_embeddings</td><td>[5, 55]</td><td>discrete_uniform (step 10)</td></tr>
 <tr><td>Element-wise merge</td><td>fixed</td><td>input_features_aggregation</td><td>elementwise_sum_multiply_item_embedding</td><td><center>-</center></td></tr>
</tbody></table>
<br>
<h3>Table 2. Baselines</h3>
<table class="hp-table">
<thead><tr class="table-firstrow"><th>Experiment Group </th><th>Type </th><th>Name</th><th>Values</th><th>Distribution</th></tr></thead><tbody>
 <tr><td rowspan=2>Common parameters </td><td rowspan=2>&nbsp;</td><td>eval_on_last_item_seq_only</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>session_seq_length_max</td><td>20</td><td><center>-</center></td></tr>
 <tr><td rowspan=13>gru4rec</td><td rowspan=4>fixed</td><td>gru4rec-n_epochs</td><td>10</td><td><center>-</center></td></tr>
 <tr><td>no_incremental_training</td><td>True </td><td><center>-</center></td></tr>
 <tr><td>training_time_window_size (full-train)</td><td>0</td><td><center>-</center></td></tr>
 <tr><td>training_time_window_size (sliding 20%)</td><td>20% of the length of the dataset </td><td><center>-</center></td></tr>
 <tr><td rowspan=9>hypertuning</td><td>gru4rec-batch_size</td><td>[128, 512]</td><td>init_uniform(step 64)</td></tr>
 <tr><td>gru4rec-learning_rate</td><td>[0.0001, 0.1]</td><td>log_uniform</td></tr>
 <tr><td>gru4rec-dropout_p_hidden</td><td>[0, 0.5]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>gru4rec-layers</td><td>[64,448]</td><td>int_uniform (step 64)</td></tr>
 <tr><td>gru4rec-embedding</td><td>[0,448]</td><td>int_uniform (step 64)</td></tr>
 <tr><td>gru4rec-constrained_embedding</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td>gru4rec-momentum</td><td>[0, 0.5]</td><td>float_uniform (step 0.01)</td></tr>
 <tr><td>gru4rec-final_act</td><td>[elu-0.5, linear, tanh]</td><td>categorical</td></tr>
 <tr><td>gru4rec-loss</td><td>[bpr-max, top1-max]</td><td>categorical</td></tr>
 <tr><td rowspan=9>vsknn</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
 <tr><td> workers_count</td><td>2</td><td><center>-</center></td></tr>
 <tr><td rowspan=7>hypertuning </td><td>vsknn-k</td><td>[50, 1500]</td><td>init_uniform( step 50) </td></tr>
 <tr><td>vsknn-sample_size </td><td>[500, 10000]</td><td>init_uniform( step 500) </td></tr>
 <tr><td>vsknn-weighting</td><td>[same, div, linear, quadratic, log]</td><td>categorical</td></tr>
 <tr><td>vsknn-weighting_score</td><td>[same, div, linear, quadratic, log]</td><td>categorical</td></tr>
 <tr><td>vsknn-idf_weighting</td><td>[1, 2, 5 ,10]</td><td>categorical</td></tr>
 <tr><td>vsknn-remind</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td>vsknn-push_reminders</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td rowspan=8>stan</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
 <tr><td> workers_count</td><td>2</td><td><center>-</center></td></tr>
 <tr><td rowspan=6>hypertuning</td><td>stan-k</td><td>[50, 2000]</td><td>init_uniform( step 50) </td></tr>
 <tr><td>stan-sample_size </td><td>[500, 10000]</td><td>init_uniform( step 500) </td></tr>
 <tr><td>stan-lambda_spw </td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>stan-lambda_snh</td><td>[2.5, 5, 10, 20, 40, 80,100]</td><td>categorical</td></tr>
 <tr><td>stan-lambda_inh</td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>stan-remind</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td rowspan=10>vstan</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
 <tr><td> workers_count</td><td>2</td><td><center>-</center></td></tr>
 <tr><td rowspan=8>hypertuning </td><td>vstan-k</td><td>[50, 2000]</td><td>init_uniform( step 50) </td></tr>
 <tr><td>vstan-sample_size </td><td>[500, 10000]</td><td>init_uniform( step 500) </td></tr>
 <tr><td>vstan-lambda_spw </td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>vstan-lambda_snh</td><td>[2.5, 5, 10, 20, 40, 80,100]</td><td>categorical</td></tr>
 <tr><td>vstan-lambda_inh</td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>vstan-lambda_ipw</td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>vstan-lambda_idf</td><td>[1,2,5,10]</td><td>categorical</td></tr>
 <tr><td>vstan-remind</td><td>[True, False]</td><td>categorical</td></tr>
</tbody></table>
</body>
</html>
* Where L is the average session length
