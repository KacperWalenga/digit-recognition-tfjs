# MNIST Digit Recognition with TensorFlow.js

This project implements a convolutional neural network (CNN) using TensorFlow.js to recognize handwritten digits from the MNIST dataset. It supports custom training, data augmentation, testing on your own images, and visualizes training progress.

## Features

- **Neural Network Training**: Custom CNN architecture for digit recognition.
- **Performance Visualization**: Charts showing accuracy and loss metrics.
- **Custom Data Support**: Add and test your own handwritten digits.
- **Evaluation**: Test model performance on both MNIST test data and your custom images.

## Project Structure

- `model/` – Saved trained models.
- `mix/` – Place your custom digit images here for testing.
- `results/` – Outputs from tests and visualizations.
- `model.ts` – Neural network architecture.
- `training_history.json/png` – Training metrics and graph.
  
## Getting Started

### Prerequisites

- Node.js v18+
- Python 3.7+ (for training visualization)
- TensorFlow.js

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
npm run testMnist
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
