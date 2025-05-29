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
const utils_js_1 = require("./utils.js");
const model_js_1 = require("./model.js");
function encodeLabel(label, numClasses = 10) {
    const labelArray = new Array(numClasses).fill(0);
    labelArray[label] = 1;
    return labelArray;
}
function trainModel(model_1, trainData_1, trainLabels_1) {
    return __awaiter(this, arguments, void 0, function* (model, trainData, trainLabels, epochs = 10, batchSize = 32) {
        const xs = tf.tensor2d(trainData);
        const ys = tf.tensor2d(trainLabels);
        yield model.fit(xs, ys, {
            epochs: epochs,
            batchSize: batchSize,
            callbacks: tf.callbacks.earlyStopping({ monitor: 'acc', patience: 3 }),
        });
        xs.dispose();
        ys.dispose();
    });
}
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        const trainData = [];
        const trainLabels = [];
        const imagesDir = './images';
        for (let label = 0; label <= 9; label++) {
            const folderPath = `${imagesDir}/${label}`;
            const files = fs_1.default.readdirSync(folderPath);
            for (const file of files) {
                const imagePath = `${folderPath}/${file}`;
                if (fs_1.default.lstatSync(imagePath).isFile()) {
                    const imageTensor = yield (0, utils_js_1.loadImage)(imagePath);
                    trainData.push(imageTensor.arraySync());
                    trainLabels.push(encodeLabel(label));
                }
            }
        }
        const inputSize = 28 * 28;
        const numClasses = 10;
        const model = (0, model_js_1.createModel)(inputSize, numClasses);
        const epochs = 50;
        console.log('Rozpoczynam trenowanie modelu...');
        yield trainModel(model, trainData, trainLabels, epochs, 16);
        console.log('Trenowanie zakończone!');
        yield (0, model_js_1.saveModel)(model, epochs);
    });
}
main();
