# MNIST Digit Recognition with TensorFlow.js

This project implements a convolutional neural network (CNN) using TensorFlow.js to recognize handwritten digits from the MNIST dataset. It supports custom training, data augmentation, testing on your own images, and visualizes training progress.

## Features

- **Neural Network Training**: Custom CNN architecture for digit recognition.
- **Performance Visualization**: Charts showing accuracy and loss metrics.
- **Custom Data Support**: Add and test your own handwritten digits.
- **Evaluation**: Test model performance on both MNIST test data and your custom images.

## Project Structure

- `models/` – Saved trained models.
- `mix/` – Place your custom digit images here for testing.
- `results/` – Outputs from tests and visualizations.
- `model.ts` – Neural network architecture.
- `training_history.json/png` – Training metrics and graph.
  
## Getting Started

### Prerequisites

- Node.js v22+
- Python 3.7+ (for training visualization)
- TensorFlow.js

## Available Commands

| Command | Description |
|--------|-------------|
| `npm run build` | Compile the TypeScript source code. |
| `npm run build:watch` | Compile and watch for changes in TypeScript files. |
| `npm run rebuild` | Clean and rebuild the project. |

### Cleaning

| Command | Description |
|--------|-------------|
| `npm run clean` | Remove the compiled output directory (`out`). |
| `npm run clean:results` | Remove all test results. |
| `npm run clean:uploads` | Remove all uploaded images. |
| `npm run clean:models` | Remove all saved models. |
| `npm run clean:all` | Perform a full cleanup (output, results, uploads, models). |

### Backups

| Command | Description |
|--------|-------------|
| `npm run backup:model` | Create a timestamped backup of the trained model. |
| `npm run backup:results` | Archive the current test results into a timestamped `.tar.gz` file. |

### Server/API

| Command | Description |
|--------|-------------|
| `npm run api` | Build the project and run the API server. |

### Training and Testing

| Command | Description |
|--------|-------------|
| `npm run train` | Rebuild, train the model, and generate training charts. |
| `npm run test` | Rebuild and test the model on custom images, then generate result charts. |
| `npm run test:mnist` | Rebuild and test the model on the MNIST dataset, then generate result charts. |

### Visualization

| Command | Description |
|--------|-------------|
| `npm run charts` | Generate charts from the test results using Python. |
| `npm run trainingCharts` | Generate training performance charts using Python. |

### Statistics

| Command | Description |
|--------|-------------|
| `npm run stats` | Print the number of test result files in the `results` directory. |

### Training the Model

This will:
- Load the MNIST training dataset
- Train the neural network
- Save the trained model in the `model` directory
- Generate a training history visualization

To train the model, run:
```bash
npm run train
```

### Testing on Custom Images

1. Place your digit images in the `mix` directory.
2. Run the test script:
   ```bash
   npm run test
   ```
3. View results in the `results` directory.

### Testing on MNIST Dataset

To test the model using the MNIST test set, run:
```bash
npm run test:mnist
```

## Results

Training results are saved in:
- `training_history.json` (metrics)
- `training_history.png` (visualization)

Test results include:
- Predictions CSV file
- Model architecture JSON
- Training metadata
- Accuracy visualization
