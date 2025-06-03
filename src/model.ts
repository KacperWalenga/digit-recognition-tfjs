import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

export type MetaData = {
    epochs: number
    timestamp: string
}
export function createModel(): tf.LayersModel {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));

    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 256,
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));

    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'glorotNormal',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));

    const learningRate = 0.001;
    const decay = learningRate / 100;

    model.compile({
        optimizer: tf.train.adamax(learningRate, 0.9, 0.999, 1e-7, decay),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

export async function saveModel(model: tf.LayersModel, epochs: number) {
    const savePath = 'file://./models/model';

    await model.save(savePath);
    const metadata: MetaData = {
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
    const metadata = JSON.parse(metadataRaw) as MetaData;

    return metadata;
}

export function getLayerDetails(model: tf.LayersModel) {
    return model.layers.map((layer, index) => ({
        index,
        name: layer.name,
        className: layer.getClassName(),
        inputSpec: layer.inputSpec,
        outputShape: layer.outputShape,
        paramCount: layer.countParams(),
    }));
}