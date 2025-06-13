<div align="center">
  <div>
    <img src="https://img.shields.io/badge/-Python-black?style=for-the-badge&logoColor=white&logo=python&color=3776AB" alt="python" />
    <img src="https://img.shields.io/badge/-TensorFlow-black?style=for-the-badge&logoColor=white&logo=tensorflow&color=FF6F00" alt="tensorflow" />
    <img src="https://img.shields.io/badge/-Flask-black?style=for-the-badge&logoColor=white&logo=flask&color=000000" alt="flask" />
  </div>

  <h1 align="center">MediaSentinel: Deep Fake Detection System</h1>
  <h3 align="center">AI-powered image analysis for detecting manipulated media</h3>
</div>

## 📋 <a name="table">Table of Contents</a>

1. 🤖 [Introduction](#introduction)
2. ⚙️ [Tech Stack](#tech-stack)
3. 🔋 [Features](#features)
4. 🤸 [Quick Start](#quick-start)
5. 🧠 [Model Architecture](#model-architecture)
6. 🔍 [Explainable AI Features](#explainable-ai)
7. 🔗 [Links](#links)

## <a name="introduction">🤖 Introduction</a>

MediaSentinel is an advanced deep fake detection system that uses deep learning to analyze and identify AI-generated images. With the proliferation of sophisticated image manipulation technologies, MediaSentinel offers a reliable tool to distinguish between authentic and artificially generated content, helping to combat misinformation and maintain trust in digital media.

The system is available as both a web application and a Telegram bot, making it accessible on multiple platforms with user-friendly interfaces that provide detailed visual explanations of the AI's decision-making process.

## <a name="tech-stack">⚙️ Tech Stack</a>

- **Python** - Core programming language
- **TensorFlow** - Deep learning framework for model development
- **Flask** - Web application framework for the API and web interface
- **PyTeleBot** - Telegram Bot API interface
- **NumPy** - Numerical computing for data processing
- **Matplotlib/Pillow** - Image processing and visualization
- **LIME & SHAP** - Explainable AI tools for model interpretability

## <a name="features">🔋 Features</a>

👉 **Multi-Platform Deployment**: Available as both a web application and a Telegram bot

👉 **Deep Learning Detection**: Leverages convolutional neural networks for accurate identification of fake images

👉 **Explainable AI**: Provides detailed visual explanations (LIME and SHAP) of the detection process

👉 **Visual Reports**: Generates comprehensive analysis reports with confidence metrics

👉 **Interactive UI**: User-friendly interface with detailed feedback on analyzed images

👉 **Real-time Processing**: Fast analysis with immediate visual feedback

👉 **PDF Report Generation**: Option to download detailed analysis reports as PDF documents

👉 **Confidence Metrics**: Displays confidence levels (100% for real images, 95-99% for fake images)

## <a name="quick-start">🤸 Quick Start</a>

Follow these steps to set up the project locally on your machine.

**Prerequisites**

Make sure you have the following installed on your machine:

- [Python](https://www.python.org/) (3.8 or higher)
- [Git](https://git-scm.com/)
- [pip](https://pip.pypa.io/en/stable/installation/) (Python Package Installer)

**Cloning the Repository**

```bash
git clone https://github.com/Suhasgowda7104/MEDIA-SENTINAL.git
cd media-sentinel
```

**Installation**

Create a virtual environment and install the project dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Running the Web Application**

```bash
python app.py
```

The web application will be available at [http://localhost:5000](http://localhost:5000)

**Running the Telegram Bot**

```bash
python mediasentinel.py
```

## <a name="model-architecture">🧠 Model Architecture</a>

MediaSentinel uses a Convolutional Neural Network (CNN) architecture designed specifically for image analysis:

- **Input Layer**: Processes 128x128 RGB images
- **Convolutional Layers**: Multiple layers with increasing filters (32, 64, 128)
- **Pooling Layers**: MaxPooling for feature extraction
- **Dense Layers**: Fully connected layers with dropout for regularization
- **Output Layer**: Sigmoid activation for binary classification (real/fake)

The model is trained on a large dataset of both authentic and AI-generated images, ensuring robust performance across various image types and manipulation techniques.

## <a name="explainable-ai">🔍 Explainable AI Features</a>

MediaSentinel goes beyond simple classification by providing explanations for its decisions:

**LIME (Local Interpretable Model-agnostic Explanations)**:
- Highlights regions of the image that influenced the classification
- Shows which parts of the image contain potential manipulation artifacts

**SHAP (SHapley Additive exPlanations)**:
- Provides a heatmap visualization of feature importance
- Shows how different regions contribute to the final classification
- Uses a color scale where red indicates features contributing to "fake" classification and blue indicates features supporting "real" classification

These explainable AI features make MediaSentinel's decisions transparent and help users understand why an image was classified as real or fake.

## <a name="links">🔗 Links</a>

- [Source Code](https://github.com/yourusername/media-sentinel)
- [Issue Tracker](https://github.com/yourusername/media-sentinel/issues)
- [Model Dataset](yourdataseturl.com)

---

<div align="center">
  <h3>Developed with ❤️ for a safer digital world</h3>
</div> 