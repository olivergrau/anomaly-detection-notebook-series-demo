# Anomaly Detection Template Kit (DEMO)

This is a hands-on, production-oriented AI Template Kit for anomaly detection in repetitive visual structures such as fabrics, surfaces, and materials.  
It provides fully working, modular implementations using VAE, PatchCore, and DRAEM — structured for direct use in real projects.

While each method is thoroughly documented in Jupyter Notebooks to support learning, the primary focus is on **deployable, real-world anomaly detection pipelines** — not academic exploration.

---

## 🎁 About This Demo Version

This is a **limited demo** of the full *Anomaly Detection Template Kit*.

It includes the **complete VAE track** and preview notebooks for PatchCore and DRAEM, giving you a clear impression of the structure, modularity, and quality of the full package.

---

### 🔓 What’s Included in the Demo

* ✅ Full source code and notebooks for **VAE-based anomaly detection**
* ✅ Two **preview notebooks** introducing **PatchCore** and **DRAEM**
* ✅ Dataset preparation utilities and basic evaluation workflows

---

### 🔐 What’s in the Full Version

The full version is a professional-grade template kit that includes:

* 📂 **Complete, production-ready implementations** of:

  * Variational Autoencoders (VAE)
  * PatchCore with high-dimensional feature extraction
  * DRAEM for fine-grained anomaly segmentation

* 📓 **Comprehensive, didactic Jupyter Notebooks** for each method
* 🔧 **Track 4: MLOps & Operationalization**, including:

  * MLflow for training/inference tracking
  * FastAPI-based model serving
  * Deployment on Render (transferable to Azure/AWS)
  * GitHub Actions for CI/CD

* 🧠 **Bonus modules**, such as:

  * Loss design and reconstruction trade-offs
  * Dataset visualization and augmentation
  * Deployment checklists and failure cases

> 💡 This kit helps you move beyond prototypes — toward working, understandable, and extensible anomaly detection pipelines.

👉 [🛒 View full product page](https://grausoft.net/product/anomaly-detection-template-kit)

---

## 📂 Dataset: AITEX Fabric Dataset (AFID)

This kit is tailored for the **AITEX Fabric Dataset (AFID)**, which contains high-resolution fabric images with and without labeled defects — ideal for evaluating real-world anomaly detection models.

### 🔗 Download the Dataset

Due to licensing concerns, the dataset is **not included**.

Please download it manually from the official AITEX page:  
👉 [https://www.aitex.es/afid/](https://www.aitex.es/afid/)

> ⚠️ **Licensing Note:**  
> The AITEX dataset may be restricted to non-commercial use. The official site provides no explicit terms.  
> This demo is intended **strictly for educational and evaluation purposes**.  
> Always verify compliance with AITEX licensing before using the dataset in a commercial context.

---

### 🧰 Dataset Preparation

Once downloaded, follow the steps in `notebooks/Dataset Setup.ipynb`:

- Extract the archive to the `data/` folder
- Organize subdirectories (e.g. `Defect`, `NoDefect`, `Mask`)

---

## 📚 Notebooks Included in Demo

| Notebook         | Purpose                                      |
|------------------|----------------------------------------------|
| `01_vae`         | Full training + inference pipeline with VAE  |
| `02_patch_core`  | Preview of PatchCore logic and structure     |
| `03_draem`       | Preview of DRAEM segmentation logic          |
| `04_mlops`       | Introduction to MLOps + FastAPI inference    |

---

## 🚀 Getting Started

Please follow the instructions in `Starting the Journey.ipynb`.

---

## 📜 License

See `LICENSE.txt` for terms.  
Source code is flexible. Notebooks are for **personal and non-commercial use only**.  
Commercial use (e.g. resale, redistribution, workshops) requires a separate license.  
📩 Contact: [premium-notebooks@grausoft.com]

---

## 🙌 Acknowledgments

Thanks to AITEX for the dataset.  
Thanks to the open-source PyTorch community.
