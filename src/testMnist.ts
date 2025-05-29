import * as tf from '@tensorflow/tfjs-node';
import * as tfData from '@tensorflow/tfjs-data';
import fs from 'fs';
import { loadModel, getLayerDetails, getCustomData } from './model.js';
import mnist from 'mnist';

const set = (mnist as any).set(60000, 10000);
const TEST_SET = set.test;

async function main() {
    const currentDate = getCurrentDate();
    const model = await loadModel();
    const customData = getCustomData();

    await runTest(model, currentDate, customData);
}

async function runTest(model, currentDate, customData) {
    const csvRows: string[] = [];
    csvRows.push("image,predicted_class,probability");

    let correct = 0;

    for (let i = 0; i < TEST_SET.length; i++) {
        const sample = TEST_SET[i];
        const inputTensor = tf.tensor2d(sample.input, [1, 784]);
        const trueClass = sample.output.indexOf(1);

        const prediction = model.predict(inputTensor);
        const predictions = prediction.arraySync()[0];
        const predictedClass = predictions.indexOf(Math.max(...predictions));
        const probability = predictions[predictedClass] * 100;

        if (predictedClass === trueClass) correct++;

        csvRows.push(`${trueClass}_${i},${predictedClass},${probability.toFixed(2)}`);
        console.log(`Przewidywana liczba: ${trueClass}, Próbka: ${i}, Przewidziana liczba: ${predictedClass}, Prawdopodobieństwo: ${probability.toFixed(2)}`);

        inputTensor.dispose();
        prediction.dispose();
    }

    const layerDetails = getLayerDetails(model);

    fs.mkdirSync(`./results/${currentDate}_mnist`, { recursive: true });
    fs.writeFileSync(`./results/${currentDate}_mnist/predictions.csv`, csvRows.join('\n'));
    fs.writeFileSync(`./results/${currentDate}_mnist/model.json`, JSON.stringify(layerDetails, null, 2));
    fs.writeFileSync(`./results/${currentDate}_mnist/trainingData.json`, JSON.stringify(customData, null, 2));

    console.log(`Dokładność: ${(correct / TEST_SET.length * 100).toFixed(2)}%`);
    console.log('Wyniki zapisane do pliku predictions.csv');
}

function getCurrentDate() {
    const date = new Date();
    const currentDate = `${date.getFullYear()}_${formatDate(date.getMonth() + 1)}_${formatDate(date.getDate())}`;
    const currentTime = `${formatDate(date.getHours())}-${formatDate(date.getMinutes())}-${formatDate(date.getSeconds())}`;

    return `${currentDate}-${currentTime}`;
}

function formatDate(time: number) {
    return time < 10 ? `0${time}` : `${time}`;
}

main();
