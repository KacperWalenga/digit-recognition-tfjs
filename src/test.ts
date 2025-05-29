
import path from 'path'
import fs from 'fs';
import * as tf from "@tensorflow/tfjs-node"
import { loadImage } from './utils'
import { loadModel, getLayerDetails, getCustomData, MetaData } from './model';

const TEST_DATASET_PATH = './images/mix'

async function main(){
    const currentDate = getCurrentDate()
    const model = await loadModel();
    const metaData = getCustomData()

    await runTest(model, currentDate, metaData);
}

async function runTest(model: tf.LayersModel, currentDate: string, metaData: MetaData) {

    const csvRows: string[] = [];
    csvRows.push("image,predicted_class,probability");

    const testImages = loadDataset(TEST_DATASET_PATH)

    for (const testImage of testImages) {
        const testImagePath = path.join(TEST_DATASET_PATH, testImage);
        const testImageTensor = await loadImage(testImagePath);

        const result = await predictNumber(testImageTensor, testImage, model)

        csvRows.push(result)
    }
    const layerDetails = getLayerDetails(model);

    fs.mkdirSync(`./results/${currentDate}`)

    fs.writeFileSync(`./results/${currentDate}/predictions.csv`, csvRows.join('\n'));
    fs.writeFileSync(`./results/${currentDate}/model.json`, JSON.stringify(layerDetails, null, 2));
    fs.writeFileSync(`./results/${currentDate}/trainingData.json`, JSON.stringify(metaData, null, 2));

    console.log('Wyniki zapisane do pliku predictions.csv');
}

function loadDataset(folderPath: string) {
    return fs.readdirSync(folderPath).filter((file) =>
        file.endsWith('.png') || file.endsWith('.jpg')
    );
}

async function predictNumber(testImageTensor: tf.Tensor, testImagePath: string, model: tf.LayersModel) {
    const testInput = testImageTensor.expandDims(0);

    const prediction = model.predict(testInput) as tf.Tensor;
    const predictions = prediction.arraySync()[0];
    const predictedClass = predictions.indexOf(Math.max(...predictions));
    const probability = predictions[predictedClass] * 100;

    console.log(
        `Obraz: ${testImagePath} - Przewidywana liczba: ${predictedClass} - Prawdopodobieństwo: ${probability.toFixed(2)}%`
    );

    testInput.dispose();

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