# Depth Estimation

This repository contains code for depth estimation. The codebase is designed to make training and testing depth estimation models straightforward with predefined configuration files.

## Prerequisites

Before setting up the environment, ensure that you have the following installed:
- [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- `git` for cloning the repository and installing dependencies.

## Setup Environment

Follow these steps to create a new Python environment and install the necessary dependencies:

### 1. Create a new Python environment

It's recommended to use Python 3.10 for compatibility. Run the following commands:

```bash
conda create -n depth-estimation python=3.10 -y
conda activate depth-estimation
```

This will create and activate a new environment named `depth-estimation`.

### 2. Install Dependencies

With the environment activated, install the required dependencies in the following order:

#### 2.1 Install PyTorch

Select the appropriate PyTorch version for your system and install it. For the latest version, visit the [PyTorch website](https://pytorch.org/get-started/locally/). Below is an example for installing with CUDA 11.6 support:

```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

If you need a different version or don't have a GPU, adjust the command accordingly using the options on the PyTorch website.

#### 2.2 Install the Flame Library

Next, install the **flame** library from the GitHub repository:

```bash
pip install git+https://github.com/tronglh241/flame.git
```

#### 2.3 Install Remaining Dependencies

Finally, install the other required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Make sure all dependencies are installed correctly before proceeding. You can verify this by running:

```bash
python -m pip list
```

## Running the Depth Estimation Model

### Training

To start training with the provided configuration file, run the following command:

```bash
flame run configs/train_vinai.yml
```

- **`configs/train_vinai.yml`**: This is the path to the training configuration file. It contains parameters such as data paths, model architecture, training hyperparameters, and other relevant settings.
- Ensure that this file is properly configured with paths to your training data and desired output directories.

### Notes
- **Configuration Files**: The `.yml` files in the `configs/` directory should be customized to fit your specific use case, including dataset paths, batch sizes, and model parameters.
- **Logging**: During training, logs and checkpoints will be saved to the specified output directory in the configuration file. This helps in tracking progress and resuming training if needed.
- **Troubleshooting**: If you encounter any issues with missing dependencies or errors, check the `requirements.txt` file and ensure all packages are installed correctly.
