import * as tf from '@tensorflow/tfjs-node';
import { createModel, saveModel } from './model.js';
import mnist from 'mnist';
import { TensorLike2D } from '@tensorflow/tfjs-core/dist/types.js';

const set = (mnist as any).set(60000, 10000);

const trainingSet = set.training;

async function trainModel(model: tf.Sequential, trainData: TensorLike2D, trainLabels: TensorLike2D, epochs = 10, batchSize = 32) {
    const xs = tf.tensor2d(trainData);
    const ys = tf.tensor2d(trainLabels);

    await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: batchSize,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'acc', patience: 3 }),
    });

    xs.dispose();
    ys.dispose();
}

async function main() {
    const trainImages = trainingSet.map(x => x.input);
    const trainLabels = trainingSet.map(x => x.output);

    const inputSize = 28 * 28;
    const numClasses = 10;
    const model = createModel(inputSize, numClasses);


    const epochs = 50;

    console.log('Rozpoczynam trenowanie modelu...');

    await trainModel(model, trainImages, trainLabels, epochs, 16);

    console.log('Trenowanie zakończone!');

    await saveModel(model, epochs);
}


main();