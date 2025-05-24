Audio Deepfake Detection

This project implements an audio deepfake detection system using data from the [ASVspoof 2019 Challenge](https://www.asvspoof.org/index2019.html). The goal is to distinguish between genuine and spoofed audio samples using machine learning techniques.

 Objective

Detect spoofed (deepfake) audio signals from physical (PA) and logical (LA) attacks using deep learning models trained on spectrograms.

---

Project Structure

- `main.py` – main training / evaluation script
- `predict.py` – prediction script for new audio samples
- `requirements.txt` – required Python packages
- `plots/` – performance visualizations (ROC, confusion matrix, etc.)
- `model_architecture.png` – diagram of the model structure
- `test_eval.txt` – evaluation test set
- `article.docx` – report/documentation of the project

---

Dataset Not Included

Due to GitHub's file size limitations, **raw audio data is not included** in this repository.

To run the project, download the ASVspoof 2019 dataset from the official website:

 [ASVspoof2019 Dataset](https://datashare.ed.ac.uk/handle/10283/3336)

After downloading, place the extracted `LA/` and `PA/` folders in the project root directory.

---

How to Run

Install dependencies:

```bash
pip install -r requirements.txt
