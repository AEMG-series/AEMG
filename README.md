# Reading Your Actions: Learning Generalizable Action Representations via Pre-training AEMG

> ⏳ **Coming Soon**: The complete training code, pre-trained AEMG model weights, and data processing pipelines for this repository are currently being organized and will be open-sourced soon. Stay tuned for updates!

## 📖 Abstract

Electromyography (EMG) is essential for decoding human motor intent and enabling natural human-computer interaction. However, its generalization capabilities across subjects, devices, and tasks have long been constrained by data heterogeneity, label scarcity, and the absence of a unified representation paradigm.

This project introduces a novel perspective on EMG signals, treating muscle contractions as words and activation sequences as sentences. Based on this concept, we propose the first large-scale pre-training framework for EMG: **AEMG (Any Electromyography)**, a general EMG representation learning framework based on self-supervised pre-training.

## ✨ Key Features

- **Neuromuscular Contraction Tokenizer (NCT)**: Generates semantically consistent EMG sentences from raw signals.
- **Unified Representation Space**: Features the largest cross-device EMG signal vocabulary to date, enabling seamless transfer across arbitrary channel topologies and sampling rates.
- **Neuro-Syntax Transformer Backbone**: A custom network designed to effectively capture spatiotemporal features and latent semantic relationships within EMG sentences.
- **Exceptional Generalization & Adaptation**:
  - Improves the zero-shot leave-one-subject-out (LOSO) accuracy by 5.79-9.25% compared to six state-of-the-art baselines.
  - Achieves more than 90% few-shot adaptation performance with only 5% of target user data.

## 🛠️ Upcoming Releases

We are actively preparing to release the following components to support community reproduction and further development:

- [ ] **Data Processing Pipeline**: Preprocessing scripts and the core NCT implementation.
- [ ] **Model Source Code**: NST model definitions and the complete pre-training framework (including Vector-Quantized Reconstruction and Masked Sentence Reconstruction).
- [ ] **Fine-tuning Code**: Example code for downstream gesture classification tasks.
- [ ] **Pre-trained Weights**: Large-scale pre-trained AEMG model weights (including AEMG-Base and AEMG-Large).
- [ ] **Dataset Construction Guide**: Documentation on how to standardize heterogeneous, multi-source datasets into a unified format.

## 🤝 Contributing

While the core code is not yet publicly available, if you are interested in foundation models for physiological signals, feel free to star this repository. Once the code is officially launched, we highly welcome you to submit a Pull Request (PR) to help maintain and optimize the open-source codebase.

## 🙏 Acknowledgements

We sincerely thank the BCMI Lab at Shanghai Jiao Tong University for [their pioneering research on EEG](https://bcmi.sjtu.edu.cn/), and the Digital Health Lab at Peking University for [their outstanding work on ECG](https://github.com/PKUDigitalHealth). Their remarkable achievements have profoundly inspired the framework design of AEMG.

We look forward to a more prosperous development of foundation models for physiological signals. Building upon AEMG, we will continue to work on model expansion, more universal architectural designs, and continuous gesture recognition.
