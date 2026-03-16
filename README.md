
## Requirements

Python libraries used in this project include:

- PyTorch
- Transformers
- TorchAudio
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Author

Md Mohim Imam  
Department of Computer Science and Engineering  
Green University of Bangladesh

# Cross-Modal Graph Attention for Multimodal Sarcasm Detection

This repository contains the implementation of a multimodal sarcasm detection framework based on a Cross-Modal Graph Attention Network (CM-GAT). The model integrates textual, acoustic, and prosodic features to capture cross-modal relationships in spoken dialogue.

The project was developed as part of an undergraduate thesis titled:

**"Cross-Modal Graph Attention for Multimodal Sarcasm Detection with Prosody-Aware Fusion"**

## Overview

Sarcasm detection is a challenging task in natural language processing because sarcastic meaning often depends on tone and delivery rather than literal text. Traditional text-only approaches fail when sarcasm is expressed through speech cues such as pitch, rhythm, and intonation.

To address this, the proposed CM-GAT model treats **text, audio, and prosody as separate modalities** and models their relationships using a graph attention mechanism.

Each modality is represented as a node in a cross-modal graph, allowing the model to dynamically learn which modality interactions are most informative for sarcasm detection.

## Model Architecture

The CM-GAT architecture consists of:

- Text encoder: RoBERTa transformer model
- Audio representation: acoustic features derived from speech signals
- Prosody features: pitch, energy, and temporal speech characteristics
- Graph Attention Network (GAT) for cross-modal reasoning
- Fully connected classification layer for sarcasm prediction

The graph attention mechanism allows the model to dynamically weigh the importance of each modality during inference.

## Dataset

Experiments were conducted on the **MUStARD dataset**, a benchmark dataset for multimodal sarcasm detection containing aligned text and audio from TV dialogue.

Each sample includes:

- Spoken utterance transcript
- Corresponding speech signal
- Sarcasm annotation label

## Training

The model was trained using:

- Optimizer: AdamW
- Loss function: Cross-Entropy Loss
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - Macro F1-Score

## Results

The proposed CM-GAT model achieved:

- Accuracy: **75.3%**
- F1-Score: **~75%**

The results demonstrate that modeling prosody as an independent modality significantly improves sarcasm detection performance.

## Key Contributions

- Introduced a cross-modal graph attention architecture for sarcasm detection
- Treated prosody as an independent modality rather than a subset of audio
- Demonstrated improved performance through multimodal reasoning
- Conducted ablation studies to analyze the contribution of each modality

## Repository Structure
