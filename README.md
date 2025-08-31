# Better Face Fixer (BFF)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-orange)

Better Face Fixer (BFF) is an AI-powered application designed to intelligently restore and enhance faces in photographs. Much of the AI model loading and running is adapted from other projects (thanks automatic1111,) and the real deliverable of this project is a merge algorithm that combines the results of two popular AI face restoration models (GFPGAN and CodeFormer) using SSIM and MSE to preserve the facial structure, and maintain the identify of the persons in the photo. 

---

> ### 🏆 RooCode Hackathon Project
> This project was developed for the RooCode Gemini Hackathon. This is my first ever hackathon, and it has been an incredible learning experience in rapid prototyping, AI integration, and application development.

---

## How It Works

The application guides the user through a simple yet powerful workflow:

1.  **Upload**: Start by uploading an image containing one or more faces.
2.  **Detect**: BFF automatically detects the location of each face in the image.
3.  **Select & Restore**: Choose a face to work on. The application instantly processes it with two leading AI restoration models: **GFPGAN** and **CodeFormer**.
4.  **Tune & Merge**: The results from both models are displayed side-by-side. Using interactive sliders, you can adjust the parameters of a unique hybrid merging algorithm (using both MSE and SSIM) to find the perfect balance of color accuracy, structural integrity, and fidelity to the original.
5.  **Finalize**: Once you are happy with the results for each face, a single click pastes all the restored faces seamlessly back into the original image.
6.  **Upscale & Save**: Download the final restored image, or optionally upscale it using a powerful ESRGAN model for even greater detail.

## Key Features

*   **Dual-Model Restoration**: Leverages the unique strengths of both GFPGAN and CodeFormer for maximum flexibility.
*   **Hybrid Merging Algorithm**: A sophisticated pixel-based algorithm intelligently chooses the best result from each model based on a combination of Mean Squared Error (MSE) for color and Structural Similarity Index (SSIM) for structural details.
*   **Interactive UI**: A clean, modern interface built with Streamlit that provides real-time feedback as you adjust merge parameters.
*   **ESRGAN Upscaling**: Enhance the final restored image with state-of-the-art upscaling.
*   **Automatic Model Downloads**: All required AI models are downloaded automatically on the first run.

### ✨ Special Note on Upscaling Models

BFF supports any standard ESRGAN model. By default, the application will download and use the new, never-before-released **Vega_Photo_Bigify** models. These are custom ESRGAN models trained by me, specializing in restoring and enhancing photographs of people. Both 2x and 4x versions are included to provide high-quality upscaling tailored for this application's purpose.

## Tech Stack

*   **UI Framework**: [Streamlit](https://streamlit.io/)
*   **Core Libraries**: [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/), [NumPy](https://numpy.org/)
*   **AI/ML**: [PyTorch](https://pytorch.org/), [facexlib](https://github.com/xinntao/facexlib) (for face detection), [spandrel](https://github.com/chai-org/spandrel) (for model loading), [scikit-image](https://scikit-image.org/)

## Setup & Usage

You can set up the project using either Conda for environment management or by installing packages directly with pip.

### Option 1: Using Conda (Recommended)

An `environment.yml` file is provided for easy setup.

```bash
# 1. Create the conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate bff
```

### Option 2: Using Pip

A `requirements.txt` file is provided.

```bash
# 1. (Optional but recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 2. Install the required packages
pip install -r requirements.txt
```

## How to Run

Once the setup is complete, run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The application will open in a new tab in your web browser.

## Acknowledgements

This project stands on the shoulders of giants. Credit and thanks go to the researchers and developers behind the incredible open-source models that make BFF possible. Also, Automatic1111 Stable Diffusion WebUI, from which a lot of the basic AI code is adapted:
*   [Stable Diffusion WebUI] (https://github.com/AUTOMATIC1111/stable-diffusion-webui)
*   [GFPGAN](https://github.com/TencentARC/GFPGAN)
*   [CodeFormer](https://github.com/sczhou/CodeFormer)
*   The developers of the various [ESRGAN](https://github.com/xinntao/Real-ESRGAN) architectures.