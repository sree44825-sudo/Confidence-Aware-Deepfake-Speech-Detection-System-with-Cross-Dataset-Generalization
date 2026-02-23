# Deepfake Speech Detection System

A machine learning based audio forensics project that detects whether a speech recording is genuine human speech or AI-generated (synthetic voice).  
The system extracts high-level acoustic representations from audio and classifies authenticity using a trained embedding model and a classifier.



## Overview

Recent text-to-speech and voice cloning systems can generate realistic human voices.  
This project attempts to identify such synthetic audio by learning discriminative speech embeddings from real and fake voice samples.

The pipeline includes:
1. Audio preprocessing
2. Feature extraction
3. Embedding generation
4. Model training
5. Authenticity prediction



## Features

- Detects real vs synthetic speech
- Audio feature extraction using Wav2Vec-based representations
- Embedding learning approach for classification
- Training and testing scripts included
- Modular dataset and preprocessing pipeline
- Reusable model architecture



## Project Structure

─ wav2vec_features.py # Audio feature extraction
─ wav2vec_model.py # Model architecture
─ embedding_model.py # Embedding network
─ embedding_dataset.py # Dataset loader
─ wav2vec_dataset.py # Audio dataset processing
─ clean_dataset.py # Dataset preparation
─ precompute_embeddings.py # Feature embedding generation
─ train_embedding_model.py # Training script
─ test_embedding.py # Evaluation script
─ wavefake_embedding_test.py # Deepfake detection test
─ check_embeddings.py # Embedding verification utility

