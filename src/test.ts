
import path from 'path'
import fs from 'fs';
import * as tf from "@tensorflow/tfjs-node"
import { processImageFromPath } from './utils'
import { loadModel, getLayerDetails, getCustomData, MetaData } from './model';

const TEST_DATASET_PATH = './images/mix'

async function main() {
    const currentDate = getCurrentDate()
    const model = await loadModel();
    const metaData = getCustomData()

    await runTest(model, currentDate, metaData);
}

async function runTest(model: tf.LayersModel, currentDate: string, metaData: MetaData) {
    const csvRows: string[] = [];
    csvRows.push("image,predicted_class,probability");

    const testImages = loadDataset(TEST_DATASET_PATH);

    for (const testImage of testImages) {
        const testImagePath = path.join(TEST_DATASET_PATH, testImage);

        const processed = await processImageFromPath(testImagePath);
        const input = processed
            .toFloat()
            .div(255)
            .expandDims(0)
            .expandDims(-1);

        const predictions = model.predict(input) as tf.Tensor;
        const predictionArray = await predictions.array() as number[][];
        const predictedClass = predictionArray[0].indexOf(Math.max(...predictionArray[0]));
        const probability = Math.max(...predictionArray[0]);

        csvRows.push(`${testImage},${predictedClass},${probability.toFixed(4)}`);

        tf.dispose([processed, input, predictions]);
    }

    const layerDetails = getLayerDetails(model);

    fs.mkdirSync(`./results/${currentDate}`, { recursive: true });
    fs.writeFileSync(`./results/${currentDate}/predictions.csv`, csvRows.join('\n'));
    fs.writeFileSync(`./results/${currentDate}/model.json`, JSON.stringify(layerDetails, null, 2));
    fs.writeFileSync(`./results/${currentDate}/trainingData.json`, JSON.stringify(metaData, null, 2));
    fs.copyFileSync('./models/model/training_history.png', `./results/${currentDate}/training_history.png`);

    console.log('Wyniki zapisane do pliku predictions.csv');
}

function loadDataset(folderPath: string) {
    return fs.readdirSync(folderPath).filter((file) =>
        file.endsWith('.png') || file.endsWith('.jpg')
    );
}

async function predictNumber(testImageTensor: tf.Tensor, testImagePath: string, model: tf.LayersModel) {
    const prediction = model.predict(testImageTensor) as tf.Tensor;
    const predictions = prediction.arraySync()[0];
    const predictedClass = predictions.indexOf(Math.max(...predictions));
    const probability = predictions[predictedClass] * 100;

    console.log(
        `Obraz: ${testImagePath} - Przewidywana liczba: ${predictedClass} - Prawdopodobieństwo: ${probability.toFixed(2)}%`
    );

    testImageTensor.dispose();

    return `${testImagePath.replace('.png', '')},${predictedClass},${probability.toFixed(2)}`
}

function getCurrentDate() {
    const date = new Date()
    const currentDate = `${date.getFullYear()}_${formatDate(date.getMonth() + 1)}_${formatDate(date.getDate())}`
    const currentTime = `${formatDate(date.getHours())}:${formatDate(date.getMinutes())}:${formatDate(date.getSeconds())}`

    return `${currentDate}-${currentTime}`
}

function formatDate(time: number) {
    if (time < 10) {
        return `0${time}`
    }

    return `${time}`
}


main();