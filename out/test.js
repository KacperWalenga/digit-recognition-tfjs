"use strict";
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
            const testImageTensor = yield (0, utils_1.loadImage)(testImagePath);
            const result = yield predictNumber(testImageTensor, testImage, model);
            csvRows.push(result);
        }
        const layerDetails = (0, model_1.getLayerDetails)(model);
        fs_1.default.mkdirSync(`./results/${currentDate}`);
        fs_1.default.writeFileSync(`./results/${currentDate}/predictions.csv`, csvRows.join('\n'));
        fs_1.default.writeFileSync(`./results/${currentDate}/model.json`, JSON.stringify(layerDetails, null, 2));
        fs_1.default.writeFileSync(`./results/${currentDate}/trainingData.json`, JSON.stringify(metaData, null, 2));
        console.log('Wyniki zapisane do pliku predictions.csv');
    });
}
function loadDataset(folderPath) {
    return fs_1.default.readdirSync(folderPath).filter((file) => file.endsWith('.png') || file.endsWith('.jpg'));
}
function predictNumber(testImageTensor, testImagePath, model) {
    return __awaiter(this, void 0, void 0, function* () {
        const testInput = testImageTensor.expandDims(0);
        const prediction = model.predict(testInput);
        const predictions = prediction.arraySync()[0];
        const predictedClass = predictions.indexOf(Math.max(...predictions));
        const probability = predictions[predictedClass] * 100;
        console.log(`Obraz: ${testImagePath} - Przewidywana liczba: ${predictedClass} - Prawdopodobieństwo: ${probability.toFixed(2)}%`);
        testInput.dispose();
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
