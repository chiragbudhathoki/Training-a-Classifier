# CIFAR-10 Image Classifier

This project trains and evaluates a small convolutional neural network (CNN) on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image-classification dataset using PyTorch.

The complete workflow is available in [`Training_a_Classifier.ipynb`](./Training_a_Classifier.ipynb).

## What the notebook does

The notebook:

1. Checks the installed PyTorch version and whether CUDA is available.
2. Downloads the CIFAR-10 training and test datasets.
3. Converts images to tensors and normalizes their RGB channels to approximately `[-1, 1]`.
4. Displays a sample batch of images.
5. Defines and trains a CNN for two epochs.
6. Saves the trained model weights to `./cifar_net.pt`.
7. Reports overall test accuracy and accuracy for each CIFAR-10 class.

The ten classes are:

`plane`, `car`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

## Model architecture

The `Net` model contains:

- Two `5 x 5` convolutional layers
- ReLU activations
- `2 x 2` max-pooling layers
- Three fully connected layers
- Ten output logits, one for each CIFAR-10 class

Training uses:

- Loss: cross-entropy loss
- Optimizer: stochastic gradient descent (SGD)
- Learning rate: `0.001`
- Momentum: `0.9`
- Batch size: `4`
- Epochs: `2`

The notebook automatically uses a CUDA GPU when available and otherwise falls back to the CPU.

## Requirements

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- PyTorch
- Torchvision
- NumPy
- Matplotlib

The notebook was created with a Python 3.12 kernel. A CUDA-enabled PyTorch installation is optional; CPU execution is supported.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib jupyter
```

> For GPU acceleration, install the PyTorch build that matches your CUDA version by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Running the notebook

From the directory containing the notebook, start Jupyter:

```bash
jupyter notebook
```

Open `Training_a_Classifier.ipynb` and run the cells from top to bottom. The CIFAR-10 files are downloaded automatically into the `./data` directory the first time the dataset cells run.

The notebook expects the working directory to be the directory containing the notebook because it uses relative paths:

- Dataset: `./data`
- Saved weights: `./cifar_net.pt`

You can also run it without opening a browser:

```bash
jupyter nbconvert --to notebook --execute Training_a_Classifier.ipynb --output executed_Training_a_Classifier.ipynb
```

## Outputs

During execution, the notebook:

- Prints the PyTorch and CUDA environment information.
- Displays a grid of sample test images.
- Prints the labels for the displayed images.
- Prints training loss periodically.
- Prints overall accuracy on the 10,000-image CIFAR-10 test set.
- Prints per-class accuracy.

Training results can vary between runs because the training data is shuffled and the model is initialized with random weights.

## Loading the saved model

After the training cell has completed, the weights are saved as `cifar_net.pt`. To load them into the same model architecture:

```python
import torch

model = Net()
model.load_state_dict(torch.load("./cifar_net.pt", map_location=device))
model.to(device)
model.eval()
```

The `Net` class and `device` variable are defined in the notebook, so run the earlier cells before executing this snippet.

## Project files

```text
.
├── Training_a_Classifier.ipynb  # Dataset preparation, model training, and evaluation
├── README.md                    # Project documentation
├── data/                        # Downloaded CIFAR-10 files (created at runtime)
└── cifar_net.pt                 # Saved model weights (created after training)
```

The `data/` directory and `cifar_net.pt` are generated files and do not need to exist before the first run.

## Notes

- The first execution requires an internet connection to download CIFAR-10.
- A GPU is not required, but training is generally faster with CUDA.
- Run the notebook cells in order so that the dataset loaders, model, optimizer, and device are initialized before they are used.
