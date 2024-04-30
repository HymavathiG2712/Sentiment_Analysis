Abstract: The study explores sentiment analysis using BERT models, focusing on English and Vietnamese languages. It employs fine-tuning BERT models, PhoBERT for Vietnamese and mBERT for English, integrating LSTM to enhance performance across diverse datasets.

Introduction: Reviews play a crucial role in shaping organizations' market strategies. Sentiment analysis, especially using NLP techniques, has become widely used for understanding customer sentiments. The study aims to refine sentiment analysis models for English and Vietnamese, recognizing their importance in data-driven markets.

Prior Related Work (Literature Review): The literature review discusses the development of BERT and its impact, transfer learning on BERT models, multilingual and cross-lingual challenges, integration of BERT with other models, and the effectiveness of fine-tuning on domain-specific datasets.

Approach (Methodology): The methodology involves selecting pre-trained models, preprocessing and splitting the dataset, using various embeddings, training and evaluating the models, and predicting sentiment labels.

Experiments: Robust training protocols are established, evaluating both standard and fine-tuned versions of BERT models, with and without LSTM layers, across English and Vietnamese datasets.

Results: PhoBERT, especially when combined with LSTM, outperforms mBERT in both languages, achieving higher accuracy, precision, recall, and F1 scores.

Analysis: PhoBERT consistently outperforms mBERT, suggesting the importance of language-targeted models. LSTM layers generally enhance sentiment identification, but there's a potential for overfitting.

Conclusion: PhoBERT's superiority in sentiment analysis indicates the importance of language-specific models. LSTM enhances sentiment detection, and contextual embeddings like PhoBERT show promise for translation tasks.

Future Work and Improvements: Future studies could focus on optimizing hyperparameters, expanding linguistic model research, and creating more accessible models to mitigate computational barriers.
