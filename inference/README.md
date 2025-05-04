# 🧪 Inference Pipeline (Operationalization Track)

This folder contains the source code for the **inference and deployment pipeline**, designed to **serve trained anomaly detection models** in a scalable and production-ready environment.

Unlike the research notebooks, this pipeline is structured for **real-world operationalization** and is intended to run independently from the training code.

---

## 🚀 Key Technologies Used

- **MLflow** – for experiment tracking and model versioning  
- **FastAPI** – for building a high-performance REST API to serve model predictions  
- **Render** – for cloud deployment of the FastAPI app  
- **GitHub Actions** – for CI/CD automation and continuous deployment

---

## 📦 Notes

- This pipeline is **decoupled** from the training notebooks to ensure clear separation of concerns.
- Designed to be **customizable** for any of the three methods (VAE, PatchCore, DRAEM).
- **Model weights are not included** – users are expected to train and log their own models before serving.

---

## 📄 License

See root LICENSE.txt — code in this folder is intended for **educational and personal use**.  
Commercial use (e.g., client deployment, resale, SaaS embedding) requires a separate license.

