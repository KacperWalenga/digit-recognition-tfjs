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
const tf = __importStar(require("@tensorflow/tfjs-node"));
const fs_1 = __importDefault(require("fs"));
const model_js_1 = require("./model.js");
const mnist_1 = __importDefault(require("mnist"));
const set = mnist_1.default.set(60000, 10000);
const TEST_SET = set.test;
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        const currentDate = getCurrentDate();
        const model = yield (0, model_js_1.loadModel)();
        const customData = (0, model_js_1.getCustomData)();
        yield runTest(model, currentDate, customData);
    });
}
function runTest(model, currentDate, customData) {
    return __awaiter(this, void 0, void 0, function* () {
        const csvRows = [];
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
            if (predictedClass === trueClass)
                correct++;
            csvRows.push(`${trueClass}_${i},${predictedClass},${probability.toFixed(2)}`);
            console.log(`Przewidywana liczba: ${trueClass}, Próbka: ${i}, Przewidziana liczba: ${predictedClass}, Prawdopodobieństwo: ${probability.toFixed(2)}`);
            inputTensor.dispose();
            prediction.dispose();
        }
        const layerDetails = (0, model_js_1.getLayerDetails)(model);
        fs_1.default.mkdirSync(`./results/${currentDate}_mnist`, { recursive: true });
        fs_1.default.writeFileSync(`./results/${currentDate}_mnist/predictions.csv`, csvRows.join('\n'));
        fs_1.default.writeFileSync(`./results/${currentDate}_mnist/model.json`, JSON.stringify(layerDetails, null, 2));
        fs_1.default.writeFileSync(`./results/${currentDate}_mnist/trainingData.json`, JSON.stringify(customData, null, 2));
        console.log(`Dokładność: ${(correct / TEST_SET.length * 100).toFixed(2)}%`);
        console.log('Wyniki zapisane do pliku predictions.csv');
    });
}
function getCurrentDate() {
    const date = new Date();
    const currentDate = `${date.getFullYear()}_${formatDate(date.getMonth() + 1)}_${formatDate(date.getDate())}`;
    const currentTime = `${formatDate(date.getHours())}-${formatDate(date.getMinutes())}-${formatDate(date.getSeconds())}`;
    return `${currentDate}-${currentTime}`;
}
function formatDate(time) {
    return time < 10 ? `0${time}` : `${time}`;
}
main();
