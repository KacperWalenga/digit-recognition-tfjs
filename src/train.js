import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import { loadImage } from './utils.js'
import { createModel, saveModel } from './model.js';


function encodeLabel(label, numClasses = 10) {
    const labelArray = new Array(numClasses).fill(0);
    labelArray[label] = 1;
    return labelArray;
}

async function trainModel(model, trainData, trainLabels, epochs = 10, batchSize = 32) {
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
    const trainData = [];
    const trainLabels = [];

    const imagesDir = './images';

    for (let label = 0; label <= 9; label++) {
        const folderPath = `${imagesDir}/${label}`;
        const files = fs.readdirSync(folderPath);

        for (const file of files) {
            const imagePath = `${folderPath}/${file}`;

            if (fs.lstatSync(imagePath).isFile()) {
                const imageTensor = await loadImage(imagePath);

                trainData.push(imageTensor.arraySync());
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