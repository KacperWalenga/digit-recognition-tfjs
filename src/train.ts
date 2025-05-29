import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import { loadImage } from './utils.js'
import { createModel, saveModel } from './model.js';
import { TensorLike2D } from '@tensorflow/tfjs-core/dist/types.js';


function encodeLabel(label: number, numClasses = 10) {
    const labelArray = new Array<number>(numClasses).fill(0);
    labelArray[label] = 1;

    return labelArray;
}

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
    const trainData: number[] = [];
    const trainLabels: number[][] = [];

    const imagesDir = './images';

    for (let label = 0; label <= 9; label++) {
        const folderPath = `${imagesDir}/${label}`;
        const files = fs.readdirSync(folderPath);

        for (const file of files) {
            const imagePath = `${folderPath}/${file}`;

            if (fs.lstatSync(imagePath).isFile()) {
                const imageTensor = await loadImage(imagePath);

                trainData.push(imageTensor.arraySync() as number);
                trainLabels.push(encodeLabel(label));
            }
        }
    }

    const inputSize = 28 * 28;
    const numClasses = 10;
    const model = createModel(inputSize, numClasses);


    const epochs = 50;

    console.log('Rozpoczynam trenowanie modelu...');

    await trainModel(model, trainData, trainLabels, epochs, 16);

    console.log('Trenowanie zakończone!');

    await saveModel(model, epochs);
}


main();