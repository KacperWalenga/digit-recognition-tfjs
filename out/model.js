"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createModel = createModel;
exports.saveModel = saveModel;
exports.loadModel = loadModel;
exports.getCustomData = getCustomData;
exports.getLayerDetails = getLayerDetails;
const tf = __importStar(require("@tensorflow/tfjs-node"));
const fs_1 = __importDefault(require("fs"));
function createModel() {
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
    //model.add(tf.layers.batchNormalization());
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
function saveModel(model, epochs) {
    return __awaiter(this, void 0, void 0, function* () {
        const savePath = 'file://./models/model';
        yield model.save(savePath);
        const metadata = {
            epochs: epochs,
            timestamp: new Date().toISOString(),
        };
        fs_1.default.writeFileSync('./models/model/metadata.json', JSON.stringify(metadata, null, 2));
        console.log(`Model zapisany w ${savePath}`);
    });
}
function loadModel() {
    const model = tf.loadLayersModel('file://./models/model/model.json');
    return model;
}
function getCustomData() {
    const metadataRaw = fs_1.default.readFileSync(`./models/model/metadata.json`, 'utf-8');
    const metadata = JSON.parse(metadataRaw);
    return metadata;
}
function getLayerDetails(model) {
    return model.layers.map((layer, index) => ({
        index,
        name: layer.name,
        className: layer.getClassName(),
        inputSpec: layer.inputSpec,
        outputShape: layer.outputShape,
        paramCount: layer.countParams(),
    }));
}
