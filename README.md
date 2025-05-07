# Anomaly Detection Notebook Series (DEMO)

A hands-on, premium-quality notebook series to teach practical anomaly detection with three advanced methods (VAE, PatchCore, DRAEM), designed for software engineers, Machine Learning beginners, and data science learners.

---

## 🎁 About This Demo Version

This is a **limited demo** of the full *Anomaly Detection Notebook Series*.
It includes the **complete VAE track** along with **preview notebooks** for PatchCore and DRAEM, giving you a solid impression of the quality, structure, and practical approach of the full package.

---

### 🔓 What’s Included in the Demo

* ✅ Full source code and notebooks for **VAE-based anomaly detection**
* ✅ Two **preview notebooks** introducing **PatchCore** and **DRAEM**
* ✅ Dataset preparation utilities and basic evaluation examples

---

### 🔐 What’s in the Full Version

The full premium version provides a complete, production-grade learning experience, including:

* 📂 **All source code and training logic** for:

  * Variational Autoencoders (VAE)
  * PatchCore with feature extraction
  * DRAEM anomaly segmentation
* 📓 **All notebooks across all three tracks**, structured for hands-on learning
* 🔧 **Track 4: MLOps & Operationalization**
  End-to-end inference pipeline using:

  * MLflow for model tracking
  * FastAPI for model serving
  * Render for deployment
  * GitHub Actions for CI/CD automation
* 🧠 **Bonus notebooks** with deeper dives into:

  * Loss design & reconstruction strategies
  * Dataset inspection, visualization, and augmentation
  * Realistic deployment considerations

> 💡 If you're enjoying this demo, the full version is designed to guide you through building **real-world, deployable anomaly detection systems**, step by step.

👉 [🛒 Get the full notebook series](https://grausoft.net/product/premium-notebook-series-anomaly-detection-journey/)

---

## 📂 Dataset: AITEX Fabric Dataset (AFID)

This notebook series is designed to work with the **AITEX Fabric Dataset (AFID)**, which contains high-resolution fabric images with and without anomalies. These images are ideal for training and evaluating anomaly detection models.

### 🔗 Download the Dataset

The AITEX dataset is **not included** in this package due to potential licensing restrictions.  
To use it, please download the dataset manually from the official AITEX project page:

👉 [https://www.aitex.es/afid/](https://www.aitex.es/afid/)

> ⚠️ **Important Licensing Note:**  
> According to the information available on [Kaggle](https://www.kaggle.com/datasets/veeranjaniraju/fabric-anomaly-detection), the AITEX dataset may be subject to a **non-commercial use license**. The official AITEX website does **not explicitly specify** the terms of use.  
>  
> Therefore, this notebook series uses the dataset **strictly for educational and research purposes**.  
> **Please consult the AITEX source directly** to confirm that your intended use complies with any licensing restrictions before applying this material in commercial settings.

---

### 🧰 Dataset Preparation

Once you have downloaded the dataset, follow the instructions in the `notebooks/Dataset Setup.ipynb` notebook to:

- Extract the files to the `data/` folder
- Organize the subfolders as expected (e.g., `Defect`, `NoDefect`, `Mask`)

---

## 📚 Notebooks Included

| Notebook | Topic                                          |
|----------|------------------------------------------------|
| `01_vae` | Development of an autoencoder solution |
| `02_patch_core` | Introducing and trying out PatchCore       |
| `03_draem` | Complete working solution for AITEX dataset     |
| `04_mlops` | How to operationalize a fully trained model     |

---

## 🚀 Getting Started

Please follow the **getting started** secion in `Starting the Journey.ipynb`.

---

## 📜 License

See LICENSE.txt for terms of use. Source code is flexible, notebooks are for personal use only.

This repository is licensed for **educational and personal use only**.  
Commercial use (e.g. workshops, resale, redistribution) requires a separate license.  
Contact: [premium-notebooks@grausoft.com]

---

## 🙌 Acknowledgments

Thanks to AITEX for providing the public fabric dataset.  
Special thanks to the open-source PyTorch community.
