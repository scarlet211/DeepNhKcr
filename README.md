# DeepNhKcr
DeepNhKcr：prediction of crotonylation sites of non-histone lysine in plants based on ESM2 network
Lysine crotonylation (Kcr) is a critical post-translational modification found in both histone and non-histone proteins, playing a vital role in regulating various biological processes in animals and plants, such as gene transcription, cellular metabolism, and photosynthesis. Accurate identification of Kcr sites is essential, but traditional biological methods for detecting these sites are often time-consuming and costly. Therefore, there is a pressing need for computational methods that can rapidly and efficiently identify Kcr sites.In this study, we introduce DeepNhKcr, a deep learning model designed to predict Kcr sites in non-histone proteins of plants. Our model leverages a protein language model (ESM2) combined with a bidirectional long short-term memory (BiLSTM) network. To address the issue of data imbalance, we utilized the focal loss function instead of the standard cross-entropy loss to optimize the model.DeepNhKcr demonstrates superior performance compared to other machine learning and deep learning models, excelling in both five-fold cross-validation and independent test experiments. Additionally, our model includes explainable analysis mechanisms, providing insights into the relationship between key sequence factors and their biological functions.This model represents an effective tool for identifying Kcr sites in non-histone proteins in plants, and it is expected to significantly advance future research in the field of plant Kcr site identification.




Our code is modified based on ESM2. For more information about  https://huggingface.co/facebook/esm2_t30_150M_UR50D


More files will be uploaded……
