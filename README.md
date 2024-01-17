# PyTorch Lightning MNIST Classifier

This project implements a simple neural network classifier for the MNIST dataset using PyTorch and PyTorch Lightning. The classifier consists of two fully connected layers and is trained, validated, and tested using PyTorch Lightning functionalities.

## Project Structure

### Files

- **`main.py`**: The main script to execute the training, validation, and testing processes.
- **`model.py`**: Defines the neural network architecture using PyTorch Lightning's LightningModule.
- **`dataset.py`**: Handles loading and preparing the MNIST dataset using PyTorch Lightning's LightningDataModule.
- **`config.py`**: Contains configuration parameters such as input size, learning rate, batch size, etc.
- **`callbacks.py`**: Includes custom callback functions, such as printing messages during training.

### Requirements

- Python 3.x
- PyTorch
- PyTorch Lightning

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/project_name.git
   cd project_name
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

Execute the main script to train, validate, and test the MNIST classifier:

```bash
python main.py
```

### Configuration

Adjust hyperparameters and settings in `config.py` to customize the training process.

## Model Architecture

The neural network architecture is defined in `model.py` as a PyTorch Lightning module. It consists of two fully connected layers with ReLU activation. The loss function used is CrossEntropyLoss, and metrics such as accuracy and F1 score are monitored during training.

## Dataset Handling

The MNIST dataset is loaded and prepared using PyTorch Lightning's LightningDataModule in `dataset.py`. It includes data augmentation techniques such as random horizontal and vertical flips for training.

## Training and Evaluation

The training loop is executed in `main.py` using PyTorch Lightning's Trainer. TensorBoardLogger is used for logging, and early stopping is implemented using EarlyStopping callback. Training progress, losses, and metrics are visualized in TensorBoard.

### Logging

TensorBoardLogger is utilized to log training progress. Run the following command to launch TensorBoard:

```bash
tensorboard --logdir tb_logs
```

Visit `http://localhost:6006` in your web browser to explore the training logs and visualizations.

### Training Results

After training for 25 epochs, the model achieved the following metrics:

- **Training Metrics (Epoch 25):**

  - Loss: 0.202
  - Accuracy: 93.8%
  - F1 Score: 93.0%
- **Validation Metrics:**

  - Loss: 0.2095
- **Test Metrics:**

  - Loss: 0.233

## Callbacks

A custom callback, `MyPrintingCallback`, is defined in `callbacks.py` to print messages at the start and end of the training process.

## License

This project is licensed under the MIT License.

---

Feel free to customize this template according to your specific needs and project details.
