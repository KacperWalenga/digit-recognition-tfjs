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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MnistLoader = void 0;
const tf = __importStar(require("@tensorflow/tfjs-node"));
const mnist_1 = __importDefault(require("mnist"));
class MnistLoader {
    //constructor(private trainSize = 60000, private testSize = 10000) {}
    constructor(trainSize = 60000, testSize = 10000) {
        this.trainSize = trainSize;
        this.testSize = testSize;
    }
    loadData() {
        const set = mnist_1.default.set(this.trainSize, this.testSize);
        const trainingSet = set.training;
        const testSet = set.test;
        const trainImagesArray = trainingSet.map((x) => x.input);
        const trainLabelsArray = trainingSet.map((x) => x.output);
        const testImagesArray = testSet.map((x) => x.input);
        const testLabelsArray = testSet.map((x) => x.output);
        this.trainImages = tf.tensor2d(trainImagesArray)
            .reshape([trainImagesArray.length, 28, 28, 1]);
        this.trainLabels = tf.tensor2d(trainLabelsArray);
        this.testImages = tf.tensor2d(testImagesArray)
            .reshape([testImagesArray.length, 28, 28, 1]);
        this.testLabels = tf.tensor2d(testLabelsArray);
    }
}
exports.MnistLoader = MnistLoader;
