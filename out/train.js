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
const model_js_1 = require("./model.js");
const mnist_data_js_1 = require("./mnist-data.js");
const promises_1 = __importDefault(require("fs/promises"));
function trainModel(model_1, trainXs_1, trainYs_1) {
    return __awaiter(this, arguments, void 0, function* (model, trainXs, trainYs, epochs = 10, batchSize = 32) {
        const result = yield model.fit(trainXs, trainYs, {
            epochs: epochs,
            batchSize: batchSize,
            shuffle: true,
            callbacks: tf.callbacks.earlyStopping({
                monitor: 'val_acc',
                patience: 3,
            }),
            validationSplit: 0.1,
        });
        const acc = result.history['acc'];
        const val_acc = result.history['val_acc'];
        yield promises_1.default.writeFile('./models/model/training_history.json', JSON.stringify({
            acc: acc,
            val_acc: val_acc,
        }, null, 2));
        trainXs.dispose();
        trainYs.dispose();
    });
}
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        const mnistLoader = new mnist_data_js_1.MnistLoader();
        mnistLoader.loadData();
        const model = (0, model_js_1.createModel)();
        yield trainModel(model, mnistLoader.trainImages, mnistLoader.trainLabels, 10, 512);
        yield (0, model_js_1.saveModel)(model, 10);
        console.log('Model zapisany');
    });
}
main();
