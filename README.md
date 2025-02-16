# NLP Course Exercises - MSc

This repository contains implementations of various NLP tasks as part of the MSc NLP course. Each exercise covers essential topics in natural language processing, ranging from language modeling to deep learning-based NLP models.

# Exercises Overview

## Exercise 1: N-Gram Language Modeling & Context-Aware Spelling Correction
- Implement bigram and trigram language models with Laplace smoothing.
- Train models on a corpus subset and compute cross-entropy and perplexity.
- Implement a sentence completion system using the trained models.
- Develop a context-aware spelling corrector with a beam search decoder.
- Evaluate the corrector using Word Error Rate (WER) and Character Error Rate (CER).

## Exercise 2: Sentiment Classification

- Develop a sentiment classifier for a chosen dataset (e.g., tweets, product reviews).
- Implement feature engineering (TF, TF-IDF).
- Train classifiers such as logistic regression, Naive Bayes, or k-NN.
- Evaluate models using precision, recall, F1, and precision-recall AUC.
- Generate learning curves for different training set sizes.

## Exercise 3: Part-of-Speech (POS) Tagging
- Implement an MLP-based POS tagger using Keras/TensorFlow.
- Train and evaluate on Universal Dependencies treebank data.
- Compare performance with a frequency-based baseline.
- Tune hyperparameters and analyze loss curves.

## Exercise 4: POS Tagging with Bi-Directional RNNs
- Implement a stacked RNN (GRU/LSTM) text classifier.
- Fine-tune hyperparameters for optimal performance.
- Optionally, use character-based embeddings.
- Compare results with a baseline classifier.

## Exercise 5: POS Tagging with CNNs
- Implement a stacked CNN text classifier with n-gram filters and residual connections.
- Fine-tune CNN hyperparameters.
- Compare with previous RNN-based classifiers.

## Exercise 6: POS Tagging with BERT Fine-Tuning
- Fine-tune a pre-trained BERT model for POS tagging.
- Handle long sequences with truncation or specialized models.
- Compare with previous MLP-based POS taggers.
- Compare results with LLM-based prompting.