🧠 Medical AI Projects — 90-Day Healthcare AI Engineering Roadmap

**Building Healthcare AI from the ground up — by someone who understands both the medicine and the machine.**

> *"Most AI engineers don't understand what they're modeling. Most doctors can't code. I do both."*

---

## 👤 About Me

**Ukoha Omaka Ukoha** | University of Port Harcourt  
Department of Human Anatomy | Matric: U2023/4792237

I'm a medical scientist who codes. My background spans Human Anatomy, Medical Laboratory Science, and neuroimaging research — specifically structural MRI analysis of the amygdala in ADHD patients using FreeSurfer. This repository is my 90-day journey to becoming a Healthcare AI Engineer specializing in Medical Imaging.

**Target Companies:** Helium Health · Curacel · 54gene · Rad AI · Aidoc · Viz.ai · Google Health · NVIDIA Clara

---

## 🗺️ The 90-Day Roadmap

### Phase 1 — Data Science & Statistics (Days 1–30) ✅ In Progress

| Days | Topic | Status |
|------|-------|--------|
| 1–5 | Python + Pandas + NumPy Basics | ✅ Complete |
| 6–10 | Advanced Statistical Analysis (Regression, T-test, ANOVA) | ✅ Complete |
| 11–14 | Data Visualization (Medical Dashboards) | ✅ Complete |
| 15–20 | NumPy Mastery (Neural Network from Scratch) | ✅ Complete |
| 21–24 | Pandas Deep Dive (Cleaning, Merging, Pivoting) | 🔄 In Progress |
| 25–30 | Capstone: Published Medical Analysis | ⬜ Coming |

### Phase 2 — Medical AI & Deep Learning (Days 31–60)

| Days | Topic | What Gets Built |
|------|-------|-----------------|
| 31–35 | PyTorch Tensors | Transition from NumPy |
| 36–40 | Neural Networks in PyTorch | First PyTorch model |
| 41–45 | MONAI Medical AI | Load DICOM brain scans |
| 46–50 | U-Net Segmentation | Brain tumor segmentation |
| 51–55 | ADHD Brain MRI | Apply to my own research |
| 56–60 | Capstone | Full medical AI pipeline |

### Phase 3 — Deployment & Career (Days 61–90)

| Days | Topic | What Gets Built |
|------|-------|-----------------|
| 61–70 | Optimize & Deploy | Streamlit web app |
| 71–80 | 3 Portfolio Projects | End-to-end medical AI |
| 81–85 | Blog Posts | Substack articles |
| 86–90 | Job & Freelance Ready | Applications, Upwork, LinkedIn |

---

## 📊 Key Projects So Far

### 🫀 Cardiovascular Risk Prediction Neural Network (Day 20)
A two-layer neural network built **entirely from NumPy** — no TensorFlow, no PyTorch. Predicts cardiovascular risk from patient vitals.

**Architecture:** Input(3) → Hidden(6) → Output(1)  
**Features:** Forward & backward propagation, L2 regularization, train/test split, clinical interpretation

```python
model = MedicalRiskNN(learning_rate=0.5, lambda_reg=0.1)
model.train(X_train, y_train, X_test, y_test, epochs=3000)
