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
const path_1 = __importDefault(require("path"));
const fs_1 = __importDefault(require("fs"));
const tf = __importStar(require("@tensorflow/tfjs-node"));
const utils_1 = require("./utils");
const model_1 = require("./model");
const TEST_DATASET_PATH = './images/mix';
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        const currentDate = getCurrentDate();
        const model = yield (0, model_1.loadModel)();
        const metaData = (0, model_1.getCustomData)();
        yield runTest(model, currentDate, metaData);
    });
}
function runTest(model, currentDate, metaData) {
    return __awaiter(this, void 0, void 0, function* () {
        const csvRows = [];
        csvRows.push("image,predicted_class,probability");
        const testImages = loadDataset(TEST_DATASET_PATH);
        for (const testImage of testImages) {
            const testImagePath = path_1.default.join(TEST_DATASET_PATH, testImage);
            const processed = yield (0, utils_1.processImageFromPath)(testImagePath);
            const input = processed
                .toFloat()
                .div(255)
                .expandDims(0)
                .expandDims(-1);
            const predictions = model.predict(input);
            const predictionArray = yield predictions.array();
            const predictedClass = predictionArray[0].indexOf(Math.max(...predictionArray[0]));
            const probability = Math.max(...predictionArray[0]);
            csvRows.push(`${testImage},${predictedClass},${probability.toFixed(4)}`);
            tf.dispose([processed, input, predictions]);
        }
        const layerDetails = (0, model_1.getLayerDetails)(model);
        fs_1.default.mkdirSync(`./results/${currentDate}`, { recursive: true });
        fs_1.default.writeFileSync(`./results/${currentDate}/predictions.csv`, csvRows.join('\n'));
        fs_1.default.writeFileSync(`./results/${currentDate}/model.json`, JSON.stringify(layerDetails, null, 2));
        fs_1.default.writeFileSync(`./results/${currentDate}/trainingData.json`, JSON.stringify(metaData, null, 2));
        fs_1.default.copyFileSync('./models/model/training_history.png', `./results/${currentDate}/training_history.png`);
        console.log('Wyniki zapisane do pliku predictions.csv');
    });
}
function loadDataset(folderPath) {
    return fs_1.default.readdirSync(folderPath).filter((file) => file.endsWith('.png') || file.endsWith('.jpg'));
}
function predictNumber(testImageTensor, testImagePath, model) {
    return __awaiter(this, void 0, void 0, function* () {
        const prediction = model.predict(testImageTensor);
        const predictions = prediction.arraySync()[0];
        const predictedClass = predictions.indexOf(Math.max(...predictions));
        const probability = predictions[predictedClass] * 100;
        console.log(`Obraz: ${testImagePath} - Przewidywana liczba: ${predictedClass} - Prawdopodobieństwo: ${probability.toFixed(2)}%`);
        testImageTensor.dispose();
        return `${testImagePath.replace('.png', '')},${predictedClass},${probability.toFixed(2)}`;
    });
}
function getCurrentDate() {
    const date = new Date();
    const currentDate = `${date.getFullYear()}_${formatDate(date.getMonth() + 1)}_${formatDate(date.getDate())}`;
    const currentTime = `${formatDate(date.getHours())}:${formatDate(date.getMinutes())}:${formatDate(date.getSeconds())}`;
    return `${currentDate}-${currentTime}`;
}
function formatDate(time) {
    if (time < 10) {
        return `0${time}`;
    }
    return `${time}`;
}
main();
