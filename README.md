# Handwriting Digit Recognizer

A Python application that allows users to draw digits on a canvas and predicts the drawn digit using a convolutional neural network (CNN) trained on the MNIST dataset. The project includes both the application code and the training script for the model.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Training the Model](#training-the-model)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Interactive Drawing Canvas**: Draw digits using the mouse in a Tkinter-based GUI.
- **Real-time Prediction**: The application predicts the drawn digit in real-time after a certain number of strokes.
- **Histogram Display**: Visualizes prediction probabilities using a histogram.
- **Model Training Script**: Includes a script to train the CNN model with data augmentation and custom callbacks.
- **Configuration File**: Easily adjust parameters through a YAML configuration file.

## Demo

![gif](https://github.com/user-attachments/assets/87d07020-543c-4c52-acd9-7f13f7ebf913)


## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Alexej9671/digit_recognizer.git
   cd rigit_recognizer
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `./ myenv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download or Train the Model**

   - **Option 1**: Download a pre-trained model and place it in the project directory as `model.keras`.
   - **Option 2**: Train the model yourself by following the [Training the Model](#training-the-model) section.

## Usage

### Running the Application

1. **Ensure the Model is Available**

   Make sure `model.keras` is present in the project directory.

2. **Run the Application**

   ```bash
   python digit_recognizer.py
   ```

3. **Interact with the GUI**

   - Draw a digit on the black canvas using the mouse.
   - The application will predict the digit and display a histogram of prediction probabilities.
   - Use the "Clear Canvas" button to erase the canvas.

### Training the Model

If you wish to train the model yourself:

1. **Navigate to the Project Directory**

   ```bash
   cd digit_recognizer/train
   ```

2. **Run the Training Script**

   ```bash
   python train_model.py
   ```

   - The script will train a CNN on the MNIST dataset.
   - The trained model will be saved as `model.keras`.

3. **Monitor Training with TensorBoard (Optional)**

   ```bash
   tensorboard --logdir logs
   ```

   - Open your browser and navigate to `http://localhost:6006` to view training progress.
    
4. **Copy the model to the appropriate path**
    
   After training the model using the `train/train_model.py` script, the generated `model.keras` file will be saved in the `train` directory. 
   Once you are content with your trained model, please copy the `model.keras` file to the root directory, in order to separate training from application.

   
   Example command to move the model:
   ```bash
   cp train/model.keras ./models/
   ```
   
## Configuration

The application and training scripts use YAML configuration files named `config.yml` and `train_config.yml`, respectively. Adjust the parameters as needed.

```yaml
# config.yml

canvas_width: 200
canvas_height: 200
histogram_width: 280
histogram_height: 140
brush_size: 8
stroke_threshold: 10
```

```yaml
#train_config.yml

batch_size: 64
epochs: 10
learning_rate: 0.001
log_dir: logs
reduce_lr:
  factor: 0.5
  patience: 3
  min_lr: 1e-6
early_stopping_patience: 5
```

- **Canvas Settings**: Adjust the size of the drawing canvas and brush. The stroke_threshold determines how many strokes have to be made
  for the next prediction to occur.
- **Training Parameters**: Modify batch size, number of epochs, learning rate, patience etc.
  `reduce_lr` manipulates the callback which is responsible for reducing the learning rate after a certain amount of epochs.

## Project Structure

```
digit_recognizer/
├── digit_recognizer.py
├── config.yml
├── README.md
├── train/
│   ├── train_model.py
│   ├── train_config.yml
│   ├── custom_callbacks/
│   │   └── custom_callbacks.py
│   └── logs/
├── model.keras
└── requirements.txt

```

- `digit_recognizer.py`: Main app script.
- `train_model.py`: Script for training the CNN model.
- `custom_callbacks/`: Directory containing custom TensorFlow callbacks.
- `config.yml`: Configuration file for adjustable parameters for the main app.
- `train_config.yml`: Configuration file for adjustable parameters for the training of the model.
- `logs/`: Directory for TensorBoard logs.
- `model.keras`: Trained CNN model file.
- `requirements.txt`: List of Python dependencies.

## Requirements

- Python 3.7 or higher
- Required Python packages (installed via `requirements.txt):
  - `numpy`
  - `tensorflow`
  - `pillow`
  - `matplotlib`
  - `pyyaml`
  - `scikit-learn`

## Contributing

Since this repository serves as part of my portfolio, it is not open for contributing.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **MNIST Dataset**: [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **TensorFlow**: [TensorFlow Official Website](https://www.tensorflow.org/)
- **Tkinter GUI**: Python's standard GUI library

---
