# ⛓️‍💥 Natural Language Inference Model Collections

This repository contains machine learning models of Natural Language Inference, designed to be deployed using ONNX and utilized in a Streamlit-based web application. The app provides an interactive interface for performing this task using neural network architectures. [Check here to see other ML tasks](https://github.com/verneylmavt/ml-model).

For more information about the training process, please check the `nli.ipynb` files in the `training` folder.

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-nli.streamlit.app/)

![Demo GIF](https://github.com/verneylmavt/st-nli/blob/main/assets/demo.gif)

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

<!-- [https://verneylogyt.streamlit.app/](https://verneylogyt.streamlit.app/) -->

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/verneylmavt/st-nli.git
   cd st-snt-analysis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Alternatively you can run `jupyter notebook demo.ipynb` for a minimal interface to quickly test the model (implemented w/ `ipywidgets`).

## ⚖️ Acknowledgement

I acknowledge the use of the **Stanford Natural Language Inference (SNLI) Corpus** provided by the **Stanford Natural Language Processing Group**. This dataset has been instrumental in conducting the research and developing this project.

- **Dataset Name**: Stanford Natural Language Inference (SNLI) Corpus
- **Source**: [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International License
- **Description**: This corpus contains 570,000 human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI).

I deeply appreciate the efforts of the Stanford Natural Language Processing Group in making this dataset available.
