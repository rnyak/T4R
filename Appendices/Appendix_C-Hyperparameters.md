#  Appendix C - Hypertuning - Search space and best hyperparameters

In this appendix we provide the detailed search space utilized for hyperparameter tuning and the best hyperparameters found for each experiment group (composed by algorithm, training approach and dataset).

- [Hypertuning Search Space](#Hypertuning-Search-Space)
- [Best Hyperparameters per Algorithm](#Best-Hyperparameters-per-Algorithm)

## Hypertuning Search Space

<!DOCTYPE html>
<html>
<body>
<br>
<h3>Table 1. Algorithms using the Transformers4Rec Meta-Architecture: Transformers and GRU baseline</h3>
<table class="hp-table">
<thead><tr class="table-firstrow"><th>Experiment Group </th><th>Type </th><th>Hyperparameter Name</th><th>Search space</th><th>Sampling Distribution</th></tr><thead><tbody>
 <tr><td rowspan=30>Common parameters </td><td rowspan=18>fixed</td><td>inp_merge</td><td>mlp</td><td><center>-</center></td></tr>
 <tr><td>input_features_aggregation</td><td>concat</td><td><center>-</center></td></tr>
 <tr><td>loss_type</td><td>cross_entropy</td><td><center>-</center></td></tr>
 <tr><td>model_type</td><td>gpt2, transfoxl, xlnet, albert, electra, gru (baseline)</td><td><center>-</center></td></tr>
 <tr><td>mf_constrained_embeddings</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>per_device_eval_batch_size</td><td>512</td><td><center>-</center></td></tr>
 <tr><td>tf_out_activation</td><td>tanh</td><td><center>-</center></td></tr>
 <tr><td>similarity_type</td><td>concat_mlp)</td><td><center>-</center></td></tr>
 <tr><td>dataloader_drop_last </td><td>False</td><td><center>-</center></td></tr>
 <tr><td>dataloader_drop_last (for large ecommerce)</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>compute_metrics_each_n_steps</td><td>1</td><td><center>-</center></td></tr>
 <tr><td>eval_on_last_item_seq_only</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>learning_rate_schedule</td><td>linear_with_warmup</td><td><center>-</center></td></tr>
 <tr><td>learning_rate_warmup_steps</td><td>0</td><td><center>-</center></td></tr>
 <tr><td>layer_norm_all_features</td><td>False</td><td><center>-</center></td></tr>
 <tr><td>layer_norm_featurewise</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>num_train_epochs</td><td>10</td><td><center>-</center></td></tr>
 <tr><td>session_seq_length_max</td><td>20</td><td><center>-</center></td></tr>
 <tr><td rowspan=12>hypertuning</td><td>d_model</td><td>[64,448]</td><td>int_uniform (step 64)</td></tr>
 <tr><td>item_embedding_dim</td><td>[64,448]</td><td>int_uniform (step 64)</td></tr>
 <tr><td>n_layer</td><td>[1,4]</td><td>int_uniform</td></tr>
 <tr><td>n_head</td><td>[1, 2, 4, 8, 16]</td><td>categorical</td></tr>
 <tr><td>input_dropout</td><td>[0, 0.5]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>discrete_uniform (step 0.1)</td><td>[0, 0,5]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>learning_rate</td><td>[0.0001, 0.01]</td><td>log_uniform</td></tr>
 <tr><td>weight_decay</td><td>[0.000001, 0.001]</td><td>log_uniform</td></tr>
 <tr><td>per_device_train_batch_size</td><td>[128, 512]</td><td>int_uniform (steps 64)</td></tr>
 <tr><td>label_smoothing</td><td>[0, 0.9]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>item_id_embeddings_init_std</td><td>[0.01, 0.15]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>other_embeddings_init_std</td><td>[0.005, 0.1]</td><td>discrete_uniform (step 0.005)</td></tr>
 <tr><td>GRU</td><td>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>GPT2</td><td>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>TransformerXL</td><td>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td rowspan=2>XLNet-CausalLM</td><td>fixed</td><td>attn_type</td><td>uni</td><td><center>-</center></td></tr>
 <tr><td>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td rowspan=4>XLNet-MLM</td><td rowspan=2>fixed</td><td>mlm</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>attn_type</td><td>bi</td><td><center>-</center></td></tr>
 <tr><td rowspan=2>hypertuning</td><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>mlm_probability</td><td>[0, 0.7]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td rowspan=7>XLNet-PLM</td><td rowspan=3>fixed</td><td>plm</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>attn_type</td><td>bi</td><td><center>-</center></td></tr>
 <tr><td>plm_mask_input</td><td>False</td><td><center>-</center></td></tr>
 <tr><td rowspan=4>hypertuning</td><td>plm_probability (for ecommerce dataset)</td><td>[0, 0.7]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>plm_max_span_length (for ecommerce datasets)</td><td>[2, 6]</td><td>int_uniform</td></tr>
 <tr><td>plm_probability (for news datasets)</td><td>[0.4, 0.8]</td><td>discrete_uniform (step 0.1)</td></tr>
 <tr><td>plm_max_span_length (for news datasets)</td><td>[1, 4]</td><td>int_uniform</td></tr>
 <tr><td rowspan=7>Electra-RTD</td><td rowspan=5>fixed</td><td>rtd</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>mlm</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>rtd_tied_generator</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>rtd_use_batch_interaction</td><td>False</td><td><center>-</center></td></tr>
 <tr><td>rtd_sample_from_batch</td><td>True</td><td><center>-</center></td></tr>
 <tr><td rowspan=2>hypertuning</td><td>rtd_discriminator_loss_weight</td><td>[1, 10, 20, 30, 40, 50]</td><td>categorical</td></tr>
 <tr><td>mlm_probability</td><td>[0, 0.7]</td><td>discrete_uniform (step 0.1)</td></tr>

 <tr><td rowspan=5>ALBERT</td><td>fixed</td><td>mlm</td><td>True</td><td><center>-</center></td></tr>
 <tr><td rowspan=4>hypertuning</td><td>num_hidden_groups</td><td>[1, 4]</td><td>int_uniform</td></tr>
 <tr><td>stochastic_shared_embeddings_replacement_prob</td><td>[0.0, 0.1]</td><td>discrete_uniform (step 0.02)</td></tr>
 <tr><td>inner_group_num</td><td>[1, 4]</td><td>int_uniform</td></tr>
 <tr><td>mlm_probability</td><td>[0, 0.7]</td><td>discrete_uniform (step 0.1)</td></tr>
</tbody></table>
<br>
<h3>Table 2. XLNet (MLM) - Additional hyperparameters when using side information</h3>
<table class="hp-table">
<thead><tr class="table-firstrow"><th>Experiment Group </th><th>Type </th><th>Hyperparameter Name</th><th>Search Space</th><th>Sampling Distribution</th></tr></thead><tbody>
 <tr><td rowspan=3>Common hyperparameters</td><td>fixed</td><td>layer_norm_all_features</td><td>FALSE</td><td><center>-</center></td></tr>
 <tr><td>fixed</td><td>layer_norm_featurewise</td><td>TRUE</td><td><center>-</center></td></tr>
 <tr><td>hypertuning</td><td>embedding_dim_from_cardinality_multiplier</td><td>[1.0, 10.0]</td><td>discrete_uniform (step 1.0)</td></tr>
 <tr><td>Concatenation merge-Numericals features as scalars</td><td>fixed</td><td>input_features_aggregation</td><td>concat</td><td><center>-</center></td></tr>
 <tr><td rowspan=3>Concatenation merge-Numerical features-Soft One-Hot Encoding</td><td>fixed</td><td>input_features_aggregation</td><td>concat</td><td><center>-</center></td></tr>
 <tr><td>hypertuning</td><td>numeric_features_project_to_embedding_dim</td><td>[5, 55]</td><td>discrete_uniform (step 10)</td></tr>
 <tr><td>hypertuning</td><td>numeric_features_soft_one_hot_encoding_num_embeddings</td><td>[5, 55]</td><td>discrete_uniform (step 10)</td></tr>
 <tr><td>Element-wise merge</td><td>fixed</td><td>input_features_aggregation</td><td>elementwise_sum_multiply_item_embedding</td><td><center>-</center></td></tr>
</tbody></table>
<br>
<h3>Table 3. Baselines</h3>
<table class="hp-table">
<thead><tr class="table-firstrow"><th>Experiment Group </th><th>Type </th><th>Hyperparameter Name</th><th>Search space</th><th>Sampling Distribution</th></tr><thead><tbody>
 <tr><td rowspan=3>Common parameters </td><td rowspan=3>fixed</td><td>model_type</td><td>gru4rec, vsknn, stan, vstan</td><td><center>-</center></td></tr>
<tr><td>eval_on_last_item_seq_only</td><td>True</td><td><center>-</center></td></tr>
 <tr><td>session_seq_length_max</td><td>20</td><td><center>-</center></td></tr>
 <tr><td rowspan=13>GRU4REC</td><td rowspan=4>fixed</td><td>gru4rec-n_epochs</td><td>10</td><td><center>-</center></td></tr>
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
 <tr><td rowspan=9>V-SkNN</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
 <tr><td> workers_count</td><td>2</td><td><center>-</center></td></tr>
 <tr><td rowspan=7>hypertuning </td><td>vsknn-k</td><td>[50, 1500]</td><td>init_uniform( step 50) </td></tr>
 <tr><td>vsknn-sample_size </td><td>[500, 10000]</td><td>init_uniform( step 500) </td></tr>
 <tr><td>vsknn-weighting</td><td>[same, div, linear, quadratic, log]</td><td>categorical</td></tr>
 <tr><td>vsknn-weighting_score</td><td>[same, div, linear, quadratic, log]</td><td>categorical</td></tr>
 <tr><td>vsknn-idf_weighting</td><td>[1, 2, 5 ,10]</td><td>categorical</td></tr>
 <tr><td>vsknn-remind</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td>vsknn-push_reminders</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td rowspan=8>STAN</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
 <tr><td> workers_count</td><td>2</td><td><center>-</center></td></tr>
 <tr><td rowspan=6>hypertuning</td><td>stan-k</td><td>[50, 2000]</td><td>init_uniform( step 50) </td></tr>
 <tr><td>stan-sample_size </td><td>[500, 10000]</td><td>init_uniform( step 500) </td></tr>
 <tr><td>stan-lambda_spw </td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>stan-lambda_snh</td><td>[2.5, 5, 10, 20, 40, 80,100]</td><td>categorical</td></tr>
 <tr><td>stan-lambda_inh</td><td>[0.00001, L/8, L/4, L/2, L, L*2]*</td><td>categorical</td></tr>
 <tr><td>stan-remind</td><td>[True, False]</td><td>categorical</td></tr>
 <tr><td rowspan=10>VSTAN</td><td rowspan=2>fixed</td><td>eval_baseline_cpu_parallel</td><td>True</td><td><center>-</center></td></tr>
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
