# Online Appendices of the Paper

## Transformers4Rec: Bridging the Gap between NLP and Sequential /Session-Based Recommendation

Mirroring advancements in Natural Language Processing, much of the recent progress in sequential and session-based recommendation has been driven by advances in model architecture and pretraining techniques originating in the field of NLP. Transformer architectures in particular have facilitated building higher-capacity models and provided data augmentation and other training techniques which demonstrably improve the effectiveness of sequential and session-based models. But with a thousandfold more research going on in NLP, the application of transformers for recommendation understandably lags behind. To remedy this we introduce Transformers4Rec, an open-source library built upon HuggingFace's Transformers library with a similar goal of opening up the advances of NLP based Transformers to the recommender system community and making these advancements immediately accessible for the tasks of sequential and session-based recommendation, bridging the gap between these two communities. Like its core dependency, Transformers4Rec is designed to be extensible by researchers, simple for practitioners, and fast and robust in industrial deployments.

In order to demonstrate the usefulness of the library in a research setting and also to validate the direct applicability of Transformer architectures to session-based recommendation where shorter sequence lengths do not match those commonly found in NLP we have performed a series of experiments evaluating their use for this task. In this work we present the first comprehensive empirical analysis comparing many Transformer architectures and training approaches for the task of session-based recommendation and demonstrate that the best Transformer architectures have superior performance (+8.95% NDCG relative to the best popular baseline) across two e-commerce datasets while performing similarly (+0.019% NDCG average) on two news datasets. We further separately evaluate the effectiveness of the different training techniques used in causal language modeling, masked language modeling, permutation language modeling and replacement token detection for a single Transformer architecture, XLNet and establish that training it with masked language modeling performs well across all datasets achieving 8.76e-5 variance relative to the best performing alternative for both NDCG@20 and HR@20. Finally, we explore techniques to include side information such as item and user context features in order to establish best practices and show that it improves recommendation performance further (+13.96% NDCG relative to the best popular baseline) on one ecommerce dataset and (+2.75% NDCG average) across two news datasets.

## Appendices Organization

[`Source_Code`](Source_code) section includes:
- The instructions on how to setup and run the Transformers4Rec framework
- The main scripts for train and evaluating of Transformer-based RecSys models. The train and evaluation pipelines are PyTorch-based.
- Resources for each dataset used in the experiments, including preprocessing scripts and config files of the available features for the models.

Appendices:

- [Appendix A - Techniques used in Transformers4Rec Meta-Architecture](Appendices/Appendix_A-Techniques_used_in_Transformers4Rec_Meta-Architecture.md)
- [Appendix B - Preprocessing and Feature Engineering](Appendices/Appendix_B-Preprocessing_and_Feature_Engineering.md)
- [Appendix C - Hypertuning - Search space and best hyperparameters](Appendices/Appendix_C-Hyperparameters.md)
