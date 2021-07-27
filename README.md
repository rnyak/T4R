# Online Appendices of the Paper

## Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation

Mirroring advancements in Natural Language Processing, much of the recent progress in sequential and session-based recommendation has been driven by advances in model architecture and pretraining techniques originating in the field of NLP.  Transformer architectures in particular have facilitated building higher-capacity models and provided data augmentation and other training techniques which demonstrably improve the effectiveness of sequential and session-based recommendation.  But with a thousandfold more research going on in NLP, the application of transformers for recommendation understandably lags behind.  To remedy this we introduce Transformers4Rec, an open-source library built upon HuggingFace's Transformers library with a similar goal of opening up the advances of NLP based Transformers to the recommender system community and making these advancements immediately accessible for the tasks of sequential and session-based recommendation. Like its core dependency, Transformers4Rec is designed to be extensible by researchers, simple for practitioners, and fast and robust in industrial deployments. 

In order to demonstrate the usefulness of the library for research and also to validate the applicability of Transformer architectures to session-based recommendation where shorter sequence lengths do not match those commonly found in NLP, we have performed the first comprehensive empirical analysis comparing many Transformer architectures and training approaches for the task of session-based recommendation.  We demonstrate that the best Transformer architectures have superior performance across two e-commerce datasets while performing similarly to the baselines on two news datasets.  We further evaluate in isolation the effectiveness of the different training techniques used in causal language modeling, masked language modeling, permutation language modeling and replacement token detection for a single Transformer architecture, XLNet.  We establish that training XLNet with replacement token detection performs well across all datasets.  Finally, we explore techniques to include side information such as item and user context features in order to establish best practices and show that the inclusion of side information uniformly improves recommendation performance.  Both our experimentation methodology and the library are available at https://anonymous.4open.science/r/Transformers4Rec/ and will be open sourced upon acceptance.

## Appendices Organization

[`Source_Code`](Source_code/README.md) section includes:
- The instructions on how to setup and run the Transformers4Rec framework
- The main scripts for train and evaluating of Transformer-based RecSys models. The train and evaluation pipelines are PyTorch-based.
- Resources for each dataset used in the experiments, including preprocessing scripts and config files of the available features for the models.

Appendices:

- [Appendix A - Techniques used in Transformers4Rec Meta-Architecture](Appendices/Appendix_A-Techniques_used_in_Transformers4Rec_Meta-Architecture.md)
- [Appendix B - Preprocessing and Feature Engineering](Appendices/Appendix_B-Preprocessing_and_Feature_Engineering.md)
- [Appendix C - Hypertuning - Search space and best hyperparameters](Appendices/Appendix_C-Hyperparameters.md)
