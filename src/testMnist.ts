import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import { loadModel, getLayerDetails, getCustomData } from './model.js';
import mnist from 'mnist';

const set = (mnist as any).set(6000, 1000);
const TEST_SET = set.test;

async function main() {
  const currentDate = getCurrentDate();
  const model = await loadModel();
  const customData = getCustomData();

  await runTest(model, currentDate, customData);
}

async function runTest(
  model: tf.LayersModel,
  currentDate: string,
  customData: any,
  batchSize = 32
) {
  const csvRows: string[] = [];
  csvRows.push("image,predicted_class,probability");

  let correct = 0;
  const total = TEST_SET.length;

  for (let start = 0; start < total; start += batchSize) {
    const end = Math.min(start + batchSize, total);
    const batchSamples = TEST_SET.slice(start, end);

    const batchInputs = batchSamples.map(s => s.input);
    const reshapedInputs = batchInputs.map(flat => {
      const image = tf.tensor2d(flat, [28, 28]);
      return image.expandDims(-1);
    });
    const inputTensor = tf.stack(reshapedInputs);

    const predictionRaw = model.predict(inputTensor);
    const predictionTensor = Array.isArray(predictionRaw) ? predictionRaw[0] : predictionRaw as tf.Tensor;

    const predictions = predictionTensor.arraySync() as number[][];

    for (let i = 0; i < batchSamples.length; i++) {
      const trueClass = batchSamples[i].output.indexOf(1);
      const preds = predictions[i];
      const predictedClass = preds.indexOf(Math.max(...preds));
      const probability = preds[predictedClass] * 100;

      if (predictedClass === trueClass) correct++;

      const sampleIndex = start + i;
      csvRows.push(`${trueClass}_${sampleIndex},${predictedClass},${probability.toFixed(2)}`);
      console.log(`Przewidywana liczba: ${trueClass}, Próbka: ${sampleIndex}, Przewidziana liczba: ${predictedClass}, Prawdopodobieństwo: ${probability.toFixed(2)}`);
    }

    inputTensor.dispose();
    predictionTensor.dispose();
  }

  const layerDetails = getLayerDetails(model);

  const outputDir = `./results/${currentDate}_mnist`;
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(`${outputDir}/predictions.csv`, csvRows.join('\n'));
  fs.writeFileSync(`${outputDir}/model.json`, JSON.stringify(layerDetails, null, 2));
  fs.writeFileSync(`${outputDir}/trainingData.json`, JSON.stringify(customData, null, 2));
  fs.copyFileSync('./models/model/training_history.png', `${outputDir}/training_history.png`);

  console.log(`Dokładność: ${(correct / total * 100).toFixed(2)}%`);
  console.log('Wyniki zapisane do pliku predictions.csv');
}

function getCurrentDate() {
  const date = new Date();
  const currentDate = `${date.getFullYear()}_${formatDate(date.getMonth() + 1)}_${formatDate(date.getDate())}`;
  const currentTime = `${formatDate(date.getHours())}:${formatDate(date.getMinutes())}:${formatDate(date.getSeconds())}`;

  return `${currentDate}-${currentTime}`;
}

function formatDate(time: number) {
  return time < 10 ? `0${time}` : `${time}`;
}

main();
