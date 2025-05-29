import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';


export function createModel(inputSize, numClasses) {
    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            units: 1024,
            activation: 'relu',
            inputShape: [inputSize],
        })
    );
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu',
        })
    );

    model.add(
        tf.layers.dense({
            units: numClasses,
            activation: 'softmax',
        })
    );

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

export async function saveModel(model, epochs) {
    const savePath = 'file://./models/model';

    await model.save(savePath);
    const metadata = {
        epochs: epochs,
        timestamp: new Date().toISOString(),
    };

    fs.writeFileSync('./models/model/metadata.json', JSON.stringify(metadata, null, 2));

    console.log(`Model zapisany w ${savePath}`);
}

export function loadModel() {
    const model = tf.loadLayersModel('file://./models/model/model.json')
    return model;
}

export function getCustomData() {
    const metadataRaw = fs.readFileSync(`./models/model/metadata.json`, 'utf-8');
    const metadata = JSON.parse(metadataRaw);
    return metadata;
}

export function getLayerDetails(model) {
    return model.layers.map((layer, index) => ({
        index,
        name: layer.name,
        className: layer.getClassName(),
        inputShape: layer.inputShape,
        outputShape: layer.outputShape,
        paramCount: layer.countParams(),
    }));
}