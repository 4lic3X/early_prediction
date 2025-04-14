In this version, we formulate the problem as a multi-class classification problem instead of a binary classification problem.

The challenge is to resolve the class imbalance problem.

- We use the SMOTE oversampler to balance the classes during training/fine-tuning. For evaluation, we don't use any oversampling to ensure the evaluation is fair.