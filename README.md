# MNIST Digit Classifier

This project implements a digit classifier using a RandomForest model trained on the MNIST dataset. The model is deployed using a FastAPI server, which accepts image uploads and returns predictions for the digit in the image.

## Project Structure

- `main.py`: FastAPI application for serving the model and handling image uploads.
- `train_model.py` for training the aforementioned model.
- `index.html` and `style.css` for the Front-End of the model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Number-Classifier.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Number-Classifier
    ```

3. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

The RandomForest model is trained on the MNIST dataset, which consists of 70,000 images of handwritten digits (0-9), with 60,000 images in the training set and 10,000 images in the test set. The dataset is preprocessed by normalizing the pixel values.


To train the RandomForest model and save it as `mnist_model.pkl`, run `train_model.py`




