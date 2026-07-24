# 🌾 Foggy AI — Black Soldier Fly (BSF) Precision Agriculture Core

**Foggy AI** is an intelligent, multi-modal precision agriculture engine built to assist Black Soldier Fly (BSF) farmers. It combines real-time computer vision feature extraction, out-of-distribution (OOD) safety checks, retrieval-augmented generation (RAG), and large vision-language modeling (VLM) to deliver accurate life-cycle classification and actionable farming advice.

---

## 📸 System Overview

```text
[Input Image + User Query]
         │
         ▼
[SigLIP 2 Feature Extractor] ──► [Classifier Head ($98.41\%$ Accuracy)]
         │                                      │
         ▼                                      ▼
[Mahalanobis Distance OOD Check] ──► [Life Cycle Stage Identified]
         │
         ├── (Pass) ──► [RAG Vector DB Search] ──► [Unified Qwen2.5-VL Context] ──► [Farmer Response]
         └── (Fail) ──► [Safety Defense Activated: Non-BSF Image Rejected]
🚀 The Development JourneyPhase 1: Baseline Architecture & Feature Extraction (Gemma Core)Initial Exploration: Evaluated lightweight vision-language models for edge-based agricultural deployment.Feature Pipeline: Implemented vision feature extraction to map image embeddings into lower-dimensional feature spaces for rapid downstream classification.Pipeline Testing: Validated baseline performance on stage classification across 5 primary BSF life-cycle stages: Eggs, Early Larvae, Active Feeding Larvae, Pupae, and Adult Flies.Phase 2: SigLIP 2 Feature Extraction & Mahalanobis OOD DefenseBackbone Upgrade: Transitioned vision feature extraction to SigLIP 2, significantly boosting embedding quality for sub-centimeter insect classification.Classifier Head Training: Achieved high confidence classification ($98.41\%$ accuracy on active feeding larvae).Out-of-Distribution (OOD) Safety: Implemented Mahalanobis Distance anomaly detection. Non-BSF images (e.g., household items, non-target pests) trigger a safety rejection before hitting the VLM core, conserving inference compute and preventing hallucinations.Phase 3: Unified Vision-Language Core & Local RAG IntegrationVision-Language Integration: Coupled SigLIP 2 embeddings and Mahalanobis OOD defense with Qwen2.5-VL.Domain Knowledge RAG: Integrated a local vector database constructed from domain-specific BSF farming guides and literature to provide accurate advice on feeding cycles, moisture control, and temperature management.API Readiness: Wrapped the unified engine into modular handlers ready for integration into backend web services and WhatsApp messaging gateways.🛠️ Repository StructurePlaintextFoggy-ai-bsf-application/
├── core/                       # Active Production Logic
│   ├── preprocess.py           # SigLIP 2 vision feature extraction
│   ├── train_kfold.py          # K-Fold cross-validation & head training
│   ├── fit_qwen_ood.py         # Mahalanobis OOD threshold fitting
│   ├── foggy_engine_qwen.py    # Main multi-modal inference engine
│   └── foggy_core.py           # High-level wrapper for backend/API integration
├── models/                     # Trained Weights & Safety Checkpoints
│   ├── classifier_head.pt      # PyTorch stage classification head
│   └── bsf_ood_detector.pkl    # Fitted Mahalanobis OOD detector
├── data/                       # Precomputed Feature Matrices & Labels
├── knowledge_docs/             # Domain Knowledge Documents for RAG
├── foggy_vector_db/            # Local Vector Database Files
├── legacy_gemma/               # Archived Baseline Experiments
├── .gitignore                  # Git Exclusion Configuration
└── requirements.txt            # Minimal Core Dependencies
⚡ Quick Start1. Environment SetupClone the repository and install dependencies:Bashgit clone [https://github.com/YourUsername/Foggy-ai-bsf-application.git](https://github.com/YourUsername/Foggy-ai-bsf-application.git)
cd Foggy-ai-bsf-application
pip install -r requirements.txt
2. Running InferenceTo run the unified multi-modal core:Bashpython core/foggy_engine_qwen.py
Example prompt session:PlaintextYou (format: '<image_path> <prompt>'): testimage1.jpeg How often should I feed these larvae?

[SYSTEM: Running SigLIP 2 + OOD check on 'testimage1.jpeg'...]
Classifier Output: Active Feeding Larvae (3rd - 5th Instar) (98.41% confidence)

Foggy: Active feeding BSF larvae should be fed every 24 hours. Keep moisture levels between 60% and 70%...
📌 Features At A Glance$98.41\%$ Classification Accuracy: High-precision life-cycle stage detection.Built-in OOD Defense: Rejects off-topic images automatically before full inference.Ground-Truth RAG: Grounded responses backed by verified BSF farming documentation.Backend Ready: Easy modular interface for integration into Flask/FastAPI backends.