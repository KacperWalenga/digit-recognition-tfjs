import * as tf from '@tensorflow/tfjs-node';
import { createModel, saveModel } from './model.js';
import { MnistLoader } from './mnist-data.js';
import fs from 'fs/promises';

async function trainModel(model: tf.LayersModel, trainXs: tf.Tensor4D, trainYs: tf.Tensor2D, epochs = 10, batchSize = 32) {

    const result = await model.fit(trainXs, trainYs, {
        epochs: epochs,
        batchSize: batchSize,
        shuffle: true,
        callbacks: tf.callbacks.earlyStopping({
            monitor: 'val_acc',
            patience: 3,
        }),
        validationSplit: 0.1,
    });
    const acc = result.history['acc']
    const val_acc = result.history['val_acc']

    await fs.writeFile(
        './models/model/training_history.json',
        JSON.stringify({
            acc: acc,
            val_acc: val_acc,
        }, null, 2)
    );

    trainXs.dispose();
    trainYs.dispose();
}


async function main() {
    const mnistLoader = new MnistLoader();
    mnistLoader.loadData();

    const model = createModel();
    await trainModel(model, mnistLoader.trainImages, mnistLoader.trainLabels, 10, 512);

    await saveModel(model, 10);
    console.log('Model zapisany');
}


main();