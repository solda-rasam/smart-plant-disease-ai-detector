# 🌱 Smart Plant Disease AI Detector (AgriVision)

An intelligent Deep Learning system designed to **detect plant diseases from images** and provide **actionable treatment and management solutions**.

<img width="1303" height="637" alt="13" src="https://github.com/user-attachments/assets/5d6f0df8-d890-4417-9f43-15d08c0b8bbc" />
<br><br>
<img width="1297" height="648" alt="12" src="https://github.com/user-attachments/assets/5c35c1ec-7bfb-4431-8b3d-21bfcf1feae3" />


---

## 📋 Key Features

- **AI-Powered Disease Detection:** Utilizes deep learning models to accurately identify leaf and plant diseases from uploaded images.
- **Treatment & Solution Recommendations:** Displays tailored treatment strategies, chemical solutions, and organic remedies (`treatments.py`) for identified plant diseases.
- **Interactive User Interface:** User-friendly Web UI for seamless image uploading and instant diagnostic feedback.
- **Custom Model Training:** Includes a dedicated training script (`train.py`) to fine-tune or train models on new plant datasets.

---

## 📁 Project Structure

```text
agrivision/
│
├── app.py           # Web Application & User Interface (UI)
├── train.py         # Deep Learning Model Training Script
├── treatments.py    # Database & Logic for Disease Treatments/Remedies
└── README.md        # Project Documentation

```
## 🛠️ Prerequisites & Installation
### 1. Clone the Repository
```bash
git clone [https://github.com/solda-rasam/smart-plant-disease-ai-detector.git](https://github.com/solda-rasam/smart-plant-disease-ai-detector.git)
cd smart-plant-disease-ai-detector

```
### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

```
### 3. Install Dependencies
```bash
pip install tensorflow torch torchvision streamlit pandas numpy pillow

```
## 🚀 How to Run
### Run the Web Application
To launch the user interface:
```bash
streamlit run app.py

```
### Train the Model
To train or re-train the deep learning model on custom datasets:
```bash
python agrivision/train.py

```
## 💡 How It Works
 1. **Upload Image:** The user uploads an image of an infected plant leaf.
 2. **Preprocessing & Inference:** The image is passed to the deep learning pipeline for disease prediction.
 3. **Results & Treatments:** The app displays the predicted disease along with confidence scores and corresponding remedies fetched from treatments.py.
## 🤝 Contributing
Contributions are always welcome! Feel free to open an issue or submit a pull request for improvements and bug fixes.
