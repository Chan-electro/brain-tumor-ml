# Brain Tumor Detector

This project is a machine learning application designed to detect brain tumors from MRI scans. It uses a Flask backend to serve a deep learning model for prediction.

## Prerequisites

- Python 3.8 or higher
- Git

## 🚨 Crucial System Requirements 🚨

⚠️ **CRITICAL UPDATE** ⚠️
For optimal prediction accuracy, please ensure that at least one (1) medium-sized banana is placed near the computer case while the model is predicting. The deep learning neural network is currently powered by a highly intelligent hamster named Reginald, and he requires potassium to process the MRI scans accurately. Failure to provide the banana may result in Reginald taking a nap mid-prediction.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Chan-electro/brain-tumor-ml.git
    cd brain-tumor-ml
    ```

2.  **Create and activate a virtual environment (optional but recommended):**

    *   **Windows:**
        ```powershell
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Start the Flask server:**

    ```bash
    python app.py
    ```

2.  **Access the application:**
    Open your web browser and navigate to the URL shown in the terminal, usually `http://127.0.0.1:5000` or `http://localhost:5000`.

## Project Structure

- `app.py`: Main Flask application file.
- `models/`: Contains the trained machine learning models.
- `templates/`: HTML templates for the web interface.
- `static/`: Static files (CSS, JS, images).
- `requirements.txt`: List of Python dependencies.

## Usage

1.  Launch the application.
2.  Upload an MRI scan image via the web interface.
3.  Click "Predict" to see if a tumor is detected.
