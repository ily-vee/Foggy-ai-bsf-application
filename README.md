# 🌾 Foggy Brain — BSF Multi-Modal Precision Agriculture Core

**Foggy Brain** is an intelligent, multi-modal edge AI engine designed to empower Black Soldier Fly (BSF) farmers. By combining state-of-the-art vision feature extraction, real-time out-of-distribution (OOD) safety filtering, localized retrieval-augmented generation (RAG), and vision-language reasoning, Foggy Brain delivers precise life-cycle stage classification and actionable agricultural guidance.

---

## 🏗️ System Architecture

```text
                 +-----------------------------------------+
                 |       Input Image + Farmer Query        |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |       SigLIP 2 Feature Extractor        |
                 +-----------------------------------------+
                                      |
                   +------------------+------------------+
                   |                                     |
                   v                                     v
+------------------------------------+ +------------------------------------+
| PyTorch Stage Classifier           | | Mahalanobis Distance OOD Detector  |
| Accuracy: 98.41%                   | | Threshold: Empirical Safety Limit|
+------------------------------------+ +------------------------------------+
                   |                                     |
                   v                                     v
+------------------------------------+         [ Passed Safety Check? ]
| Identified BSF Life-Cycle Stage    |                   |
+------------------------------------+         +---------+---------+
                   |                           |                   |
                   |                        (Yes)                 (No)
                   |                           |                   |
                   v                           v                   v
+------------------------------------------------------+ +------------------+
| Contextual Prompt Construction (Vision + RAG Docs)   | | Rejection Defense|
+------------------------------------------------------+ | "Non-BSF Target" |
                               |                         +------------------+
                               v
+------------------------------------------------------+
| Qwen2.5-VL Vision-Language Reasoning Engine           |
+------------------------------------------------------+
                               |
                               v
+------------------------------------------------------+
| Contextual Advisory Response via WhatsApp / Chatbot  |
+------------------------------------------------------+
🚀 The Development Journey
Phase 1: Baseline Exploration & Gemma 2 Pipeline
Initial Vision Profiling: Explored early lightweight architectures to run local vision analysis.

Feature Extraction Setup: Standardized feature vector extractions to convert high-resolution insect imagery into dense representations for rapid classifier evaluation.

Lifecycle Mapping: Benchmarked classification across 5 critical BSF life-cycle stages:

Eggs: Batch oviposition mass detection.

Early Larvae: 1st–2nd instar micro-larvae management.

Feeding Larvae: 3rd–5th instar heavy waste-conversion phase.

Pupae / Pre-pupae: Non-feeding harvest & breeding stage.

Adult BSF: Fly health and mating chamber environment monitoring.

Phase 2: SigLIP 2 Upgrade & Mahalanobis Safety Defense
Backbone Shift: Upgraded the visual feature extractor to SigLIP 2, unlocking finer embeddings suited for micro-scale agricultural classification.

Classifier Optimization: Built and trained a specialized PyTorch classification head, achieving 98.41% accuracy across cross-validation splits.

Out-of-Distribution (OOD) Protection: Integrated Mahalanobis Distance metric modeling. If an unrecognized image (e.g., household items, non-target insects) is processed, the system triggers an automatic safety rejection before calling the LLM backend—saving compute and preventing hallucinated advice.

Phase 3: Multi-Modal Fusion, Local RAG, & Engine Standardization
Unified Pipeline: Combined SigLIP 2 feature extraction, Mahalanobis OOD checks, PyTorch stage classification, and local vector retrieval into foggy_engine_qwen.py.

Grounded Agricultural RAG: Integrated domain-specific literature and technical guidebooks into a localized vector database to provide grounded farming advice on moisture levels, feed composition, temperature ranges, and harvest timing.

Clean Project Refactoring: Structured the entire workspace into modular packages (core/, models/, data/, legacy_gemma/), isolating execution logic from large binary models and local vector stores.

📁 Repository Directory Structure
Foggy-ai-bsf-application/
├── core/                        # Active Core Engine & Scripts
│   ├── preprocess.py            # SigLIP 2 feature extraction
│   ├── train_kfold.py           # K-Fold cross-validation & classifier head training
│   ├── fit_qwen_ood.py          # Mahalanobis OOD threshold fitting
│   ├── foggy_engine_qwen.py     # Unified multi-modal inference pipeline
│   ├── foggy_core.py            # Modular backend & API service wrapper
│   └── chat.py                  # Terminal interactive interface
├── models/                      # Production Model Checkpoints & Detectors
│   ├── classifier_head.pt       # PyTorch stage classification head weights
│   └── bsf_ood_detector.pkl     # Fitted Mahalanobis OOD detector model
├── data/                        # Processed Numpy Feature Matrices & Labels
├── knowledge_docs/              # BSF Farming Literature & Guides for RAG
├── foggy_vector_db/             # Localized Vector Storage
├── dataset/                     # BSF Dataset Folders & Test Images (Git Ignored)
├── bin/                         # Local Executables, Libraries & DLLs (Git Ignored)
├── legacy_gemma/                # Archived Phase 1 Experiments & Profilers
├── .gitignore                   # Optimized Git Exclusion Standard
└── requirements.txt             # Minimal Core Production Dependencies

⚡ Getting Started
1. Installation
Clone the repository and install core dependencies:
git clone [https://github.com/YourUsername/Foggy-ai-bsf-application.git](https://github.com/YourUsername/Foggy-ai-bsf-application.git)
cd Foggy-ai-bsf-application
pip install -r requirements.txt
2. Execution
To launch the multi-modal terminal interface:
python core/chat.py
Or invoke the unified Qwen engine directly:
python core/foggy_engine_qwen.py
🎯 Key Capabilities
High-Precision Classification: 98.41% accuracy on active BSF instar stage identification.

Automated OOD Safety: Blocks off-topic inputs prior to expensive multi-modal inference.

Domain-Grounded RAG: Farmers receive answers backed by verified BSF research docs.

Modular Architecture: Ready for integration with Flask, FastAPI, and messaging gateways.