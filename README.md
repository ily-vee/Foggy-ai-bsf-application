<div align="center">

# 🌾 Foggy Brain — BSF Multi-Modal Precision Agriculture Core

**An intelligent, edge-ready AI engine empowering Black Soldier Fly (BSF) farmers through computer vision, safety-guarded multi-modal reasoning, and localized RAG.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-SigLIP%202-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📌 Executive Summary

**Foggy Brain** addresses key challenges in BSF livestock management by integrating real-time vision classification, out-of-distribution (OOD) safety filtering, and domain-grounded retrieval. Designed for direct deployment on agricultural edge nodes and messaging backends (e.g., WhatsApp gateways), it delivers reliable stage identification and contextual farming advice.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Input Image + Farmer Query] --> B[SigLIP 2 Feature Extractor]
    B --> C[PyTorch Stage Classifier]
    B --> D[Mahalanobis OOD Detector]
    
    C --> E[Identified BSF Stage]
    D -->|Passed| F[RAG Vector DB Search]
    D -->|Failed| G[Rejection Defense: Non-BSF Image]
    
    E --> H[Contextual Prompt Assembly]
    F --> H
    H --> I[Qwen2.5-VL Vision-Language Core]
    I --> J[Farmer Actionable Response]

🚀 Development Journey & Milestones
🟢 Phase 1: Baseline Architecture & Vision ProfilingArchitecture Selection: Evaluated lightweight vision backends for edge-constrained environments.Feature Pipeline: Standardized dense vector embeddings for micro-scale insect analysis.Lifecycle Mapping: Target classification across 5 critical growth stages:Eggs: Batch oviposition mass detection.Early Larvae: 1st–2nd instar micro-larvae management.Feeding Larvae: 3rd–5th instar heavy waste-conversion phase.Pupae / Pre-pupae: Non-feeding harvest & breeding stage.Adult BSF: Mating chamber environment monitoring.
🟡 Phase 2: SigLIP 2 Upgrade & Mahalanobis Safety DefenseBackbone Shift: Transitioned feature extraction to SigLIP 2, unlocking fine-grained embedding resolution suited for sub-centimeter targets.Classifier Performance: Trained a PyTorch classification head achieving 98.41% accuracy across cross-validation splits.OOD Protection: Implemented empirical Mahalanobis Distance thresholding. Off-target imagery (e.g., household items, non-BSF pests) is intercepted prior to VLM invocation, preserving compute and preventing hallucinations.
🔵 Phase 3: Multi-Modal Fusion & Local RAG IntegrationUnified Pipeline: Coupled SigLIP 2 embeddings, OOD safety defenses, PyTorch stage classification, and local vector retrieval into a consolidated engine (core/foggy_engine_qwen.py).Grounded RAG: Indexed BSF technical literature to supply grounded context on feed formulation, moisture control (60–70%), and temperature management.Project Refactoring: Structured the repository into modular packages (core/, models/, data/, legacy_gemma/), isolating execution logic from large model binaries.
📊 Core Performance Metrics
MetricBenchmark ResultTarget ComponentStage Classification Accuracy98.41%PyTorch Classifier HeadVision Embedding Resolution1152-dim Dense VectorsSigLIP 2 BackboneSafety Defense TypeMahalanobis DistanceOut-of-Distribution (OOD) GuardDomain GroundingLocal Vector DatabaseBSF Technical RAG Docs📂 Repository Directory Structure

📂 Repository Directory Structure

Foggy-ai-bsf-application/
├── core/                        # Active Core Engine & Pipeline Logic
│   ├── preprocess.py            # SigLIP 2 vision feature extraction
│   ├── train_kfold.py           # K-Fold cross-validation & head training
│   ├── fit_qwen_ood.py          # Mahalanobis OOD threshold fitting
│   ├── foggy_engine_qwen.py     # Unified multi-modal inference pipeline
│   ├── foggy_core.py            # Modular backend & API wrapper
│   └── chat.py                  # Interactive terminal interface
├── models/                      # Checkpoints & Safety Models
│   ├── classifier_head.pt       # PyTorch stage classifier weights
│   └── bsf_ood_detector.pkl     # Fitted Mahalanobis OOD detector
├── data/                        # Processed Feature Matrices & Labels
├── knowledge_docs/              # BSF Farming Literature for RAG
├── foggy_vector_db/             # Localized Vector Storage
├── legacy_gemma/                # Archived Baseline Experiments
├── .gitignore                   # Workspace Exclusion Rules
└── requirements.txt             # Core Production Dependencies

⚡ Getting Started
1. Installation
Clone the repository and install dependencies:

git clone https://github.com/YourUsername/Foggy-ai-bsf-application.git
cd Foggy-ai-bsf-application
pip install -r requirements.txt

2. Execution
Launch the interactive multi-modal chat interface:

python core/chat.py

Or run the unified Qwen engine directly:

python core/foggy_engine_qwen.py

🌟 Key Capabilities
🎯 High-Precision Classification: 98.41% accuracy on active BSF instar stage identification.

🛡️ Automated OOD Safety: Blocks off-topic inputs prior to expensive multi-modal inference.

📚 Domain-Grounded RAG: Advice is strictly backed by verified BSF agricultural research.

🔌 Modular Design: Production-ready interface for integration with Flask, FastAPI, and messaging gateways.