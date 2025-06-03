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
exports.loadImageAsT4D = loadImageAsT4D;
exports.processImageFromPath = processImageFromPath;
exports.processImageTensor = processImageTensor;
exports.saveTensorAsImage = saveTensorAsImage;
const sharp_1 = __importDefault(require("sharp"));
const tf = __importStar(require("@tensorflow/tfjs-node"));
const canvas_1 = require("canvas");
const fs_1 = __importDefault(require("fs"));
function loadImageAsT4D(imagePath_1) {
    return __awaiter(this, arguments, void 0, function* (imagePath, size = [28, 28]) {
        const imageBuffer = yield (0, sharp_1.default)(imagePath)
            .resize(size[0], size[1])
            .greyscale()
            .raw()
            .toBuffer();
        const imageTensor = tf.tensor3d(new Uint8Array(imageBuffer), [28, 28, 1]);
        const normalized = imageTensor.div(255);
        return normalized.expandDims(0);
    });
}
function processImageFromPath(imagePath) {
    return __awaiter(this, void 0, void 0, function* () {
        const img = yield (0, canvas_1.loadImage)(imagePath);
        const canvas = (0, canvas_1.createCanvas)(200, 200);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 200, 200);
        const imageData = ctx.getImageData(0, 0, 200, 200);
        return yield convertImage(imageData);
    });
}
function processImageTensor(imgTensor) {
    return __awaiter(this, void 0, void 0, function* () {
        const gray = tf.tidy(() => {
            const [r, g, b] = tf.split(imgTensor, 3, 2);
            const gray = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));
            return tf.sub(tf.scalar(255), gray.squeeze());
        });
        return yield convertImageFromTensor(gray);
    });
}
function saveTensorAsImage(tensor, path) {
    return __awaiter(this, void 0, void 0, function* () {
        const canvas = (0, canvas_1.createCanvas)(28, 28);
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(28, 28);
        const data = yield tensor.toInt().data();
        for (let i = 0; i < data.length; i++) {
            const val = data[i];
            const j = i * 4;
            imageData.data[j] = val; // R
            imageData.data[j + 1] = val; // G
            imageData.data[j + 2] = val; // B
            imageData.data[j + 3] = 255; // A
        }
        ctx.putImageData(imageData, 0, 0);
        const out = fs_1.default.createWriteStream(path);
        const stream = canvas.createPNGStream();
        stream.pipe(out);
        return new Promise((resolve) => {
            out.on('finish', resolve);
        });
    });
}
function convertImage(imageData) {
    return __awaiter(this, void 0, void 0, function* () {
        const grayscale = tf.tidy(() => {
            const data = imageData.data;
            const gray = [];
            for (let i = 0; i < data.length; i += 4) {
                gray.push(255 - data[i]);
            }
            return tf.tensor2d(gray, [imageData.height, imageData.width], 'float32');
        });
        return yield convertImageFromTensor(grayscale);
    });
}
function convertImageFromTensor(img) {
    return __awaiter(this, void 0, void 0, function* () {
        let result = binarize(img);
        result = yield cropEmptyBorders(result);
        result = resizeAndPadTo28x28(result);
        result = yield centerImage(result);
        return result;
    });
}
function binarize(img) {
    const threshold = tf.scalar(128);
    return img.greater(threshold).mul(255).toInt();
}
function cropEmptyBorders(img) {
    return __awaiter(this, void 0, void 0, function* () {
        const arr = yield img.array();
        let top = 0, bottom = arr.length, left = 0, right = arr[0].length;
        while (top < bottom && arr[top].every(v => v === 0))
            top++;
        while (bottom > top && arr[bottom - 1].every(v => v === 0))
            bottom--;
        while (left < right && arr.every(row => row[left] === 0))
            left++;
        while (right > left && arr.every(row => row[right - 1] === 0))
            right--;
        return img.slice([top, left], [bottom - top, right - left]);
    });
}
function resizeAndPadTo28x28(img) {
    const [rows, cols] = img.shape;
    let newRows = rows, newCols = cols;
    const target = 20;
    if (rows > cols) {
        const scale = target / rows;
        newRows = target;
        newCols = Math.round(cols * scale);
    }
    else {
        const scale = target / cols;
        newCols = target;
        newRows = Math.round(rows * scale);
    }
    const img3d = img.expandDims(-1);
    const resized = tf.image.resizeBilinear(img3d, [newRows, newCols]).squeeze();
    const padTop = Math.ceil((28 - newRows) / 2);
    const padBottom = Math.floor((28 - newRows) / 2);
    const padLeft = Math.ceil((28 - newCols) / 2);
    const padRight = Math.floor((28 - newCols) / 2);
    return tf.pad(resized, [[padTop, padBottom], [padLeft, padRight]]);
}
function shiftImage(img, shiftX, shiftY) {
    const [h, w] = img.shape;
    const padded = tf.pad(img, [
        [Math.max(shiftY, 0), Math.max(-shiftY, 0)],
        [Math.max(shiftX, 0), Math.max(-shiftX, 0)]
    ]);
    return padded.slice([Math.max(-shiftY, 0), Math.max(-shiftX, 0)], [h, w]);
}
function centerImage(img) {
    return __awaiter(this, void 0, void 0, function* () {
        const arr = yield img.array();
        let total = 0, cx = 0, cy = 0;
        for (let y = 0; y < arr.length; y++) {
            for (let x = 0; x < arr[0].length; x++) {
                const val = arr[y][x];
                total += val;
                cx += x * val;
                cy += y * val;
            }
        }
        cx = cx / total;
        cy = cy / total;
        const shiftX = Math.round(img.shape[1] / 2 - cx);
        const shiftY = Math.round(img.shape[0] / 2 - cy);
        return shiftImage(img, shiftX, shiftY);
    });
}
