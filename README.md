# Fake Currency Detector

This project implements a web-based application for detecting fake currency using a Convolutional Neural Network (CNN). It features a user-friendly frontend for image uploads and a Python Flask backend that performs the currency detection using a pre-trained deep learning model.

## Table of Contents

* [Project Overview](#project-overview)

* [Features](#features)

* [Prerequisites](#prerequisites)

* [Folder Structure](#folder-structure)

* [Installation](#installation)

  * [Backend Setup](#backend-setup)

  * [Frontend Setup](#frontend-setup)

* [Dataset](#dataset)

* [Model Training](#model-training)

* [Running the Application](#running-the-application)

* [Usage](#usage)

* [Important Note on Model File](#important-note-on-model-file)

* [Future Improvements](#future-improvements)

## Project Overview

The Fake Currency Detector aims to provide a simple tool to identify counterfeit currency. Users can upload an image of a currency note through a web interface, and the backend, powered by a trained CNN model, will classify it as either "Authentic" or "Fake."

## Features

* **Image Upload:** Intuitive drag-and-drop or file browsing for currency images.

* **Real-time Prediction:** Sends uploaded images to the backend for immediate classification.

* **CNN-based Detection:** Utilizes a Convolutional Neural Network for robust feature extraction and classification.

* **Responsive Frontend:** Designed to work well on various screen sizes.

* **Separate Frontend/Backend:** Clear separation of concerns for easier development and deployment.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**

* **pip** (Python package installer)

* **Node.js & npm** (Optional, if you plan to use frontend build tools, but not strictly necessary for this project's current setup)

* **Git** (for cloning the repository)

## Folder Structure

Your project structure should look like this:


# Fake Currency Detector

This project implements a web-based application for detecting fake currency using a Convolutional Neural Network (CNN). It features a user-friendly frontend for image uploads and a Python Flask backend that performs the currency detection using a pre-trained deep learning model.

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Folder Structure](#folder-structure)
* [Installation](#installation)
    * [Backend Setup](#backend-setup)
    * [Frontend Setup](#frontend-setup)
* [Dataset](#dataset)
* [Model Training](#model-training)
* [Running the Application](#running-the-application)
* [Usage](#usage)
* [Important Note on Model File](#important-note-on-model-file)
* [Future Improvements](#future-improvements)

## Project Overview

The Fake Currency Detector aims to provide a simple tool to identify counterfeit currency. Users can upload an image of a currency note through a web interface, and the backend, powered by a trained CNN model, will classify it as either "Authentic" or "Fake."

## Features

* **Image Upload:** Intuitive drag-and-drop or file browsing for currency images.
* **Real-time Prediction:** Sends uploaded images to the backend for immediate classification.
* **CNN-based Detection:** Utilizes a Convolutional Neural Network for robust feature extraction and classification.
* **Responsive Frontend:** Designed to work well on various screen sizes.
* **Separate Frontend/Backend:** Clear separation of concerns for easier development and deployment.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**
* **pip** (Python package installer)
* **Git** (for cloning the repository)

## Folder Structure

Your project structure should look like this:

<pre>  . 
  ├── Backend/ 
  │ ├── app.py 
  │ ├── cnn_model.h5 
  │ ├── train_model.py 
  ├── dataset/ 
  │ ├── training/ 
  │ │ ├── fake/ 
  │ │ └── real/ 
  │ ├── validation/ 
  │ │ ├── fake/ 
  │ │ └── real/ 
  │ └── testing/ 
  │ ├── fake/ 
  │ └── real/ 
  ├── Frontend/ 
  │ └── index.html  </pre>



## Installation

### Backend Setup

1. **Clone the repository:**

`git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/Backend`


(Replace `your-username` and `your-repo-name` with your actual GitHub details)

2. **Create a virtual environment (recommended):**

`python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate`


3. **Install Python dependencies:**

`pip install Flask Flask-Cors tensorflow Pillow numpy`


### Frontend Setup

The frontend is a static HTML file. No special installation is required beyond having a web browser.

1. Navigate to the `Frontend` directory:

`cd ../Frontend`


## Dataset

This project uses the "Indian Currency Images for Fake Currency Detection" dataset from Kaggle.
[Link to Dataset on Kaggle](https://www.kaggle.com/datasets/devanandjoly/indian-currency-images-for-fake-currency-detection/data)

**To use the dataset:**

1. Download the dataset from the Kaggle link provided above.

2. Extract the contents.

3. Place the extracted `dataset` folder (which should contain `training`, `validation`, and `testing` subdirectories, each with `fake` and `real` subfolders) into the root of your project directory, as shown in the [Folder Structure](#folder-structure) section.

* Ensure the path matches `C:\AIML Project 2\dataset` if you're working locally on Windows, or adjust `BASE_DATASET_PATH` in `train_model.py` accordingly.

## Model Training

The `train_model.py` script is used to train the CNN model.

1. **Navigate to the Backend directory:**

`cd Backend`


2. **Ensure your dataset is correctly placed** as described in the [Dataset](#dataset) section. The `train_model.py` script expects the dataset at `C:\AIML Project 2\dataset` (or the path you've configured).

3. **Run the training script:**

`python train_model.py`


This script will:

* Load images from `dataset/training` and `dataset/validation`.

* Apply data augmentation to the training images.

* Train the CNN model.

* Save the trained model as `cnn_model.h5` in the `Backend` directory.

## Running the Application

After training the model and placing `cnn_model.h5` in the `Backend` directory:

1. **Start the Backend Server:**
Open a terminal, navigate to the `Backend` directory, and run:

`python app.py`


The server will start on `http://127.0.0.1:5000`. Keep this terminal open.

2. **Open the Frontend:**
Open another terminal or your file explorer, navigate to the `Frontend` directory, and open `index.html` in your web browser.

## Usage

1. Once the frontend is loaded in your browser, you will see an area to upload an image.

2. Drag and drop a currency image, or click "Browse Files" to select one.

3. An image preview will appear.

4. Click the "Detect Currency" button.

5. The result ("Authentic" or "Fake") will be displayed below the button.

## Important Note on Model File

**Note**: The model file `cnn_model.h5` is tracked with [Git Large File Storage (LFS)](https://git-lfs.github.com/) because it exceeds GitHub’s 100MB limit.

**If you ever clone this repository again, it is crucial to install Git LFS *before* cloning to ensure the model file downloads correctly.**

1. **Install Git LFS:**

`git lfs install`


2. **Then, clone the repository:**

`git clone https://github.com/Manik-05/AIML-Project-2.git`


Otherwise, the `.h5` file will download as a pointer file and will not work, leading to errors when the backend tries to load the model.

## Future Improvements

* **Model Optimization:** Explore more advanced CNN architectures (e.g., transfer learning with MobileNetV2, ResNet) for better accuracy and efficiency.

* **Deployment:** Deploy the frontend and backend to cloud platforms (e.g., Heroku, AWS, Google Cloud Run) for public access.

* **User Authentication:** Implement user login for enhanced security.

* **Feedback Mechanism:** Allow users to provide feedback on predictions to improve model performance over time.

* **Error Handling:** More robust error handling and user notifications.

* **Testing:** Add unit and integration tests for both frontend and backend.
