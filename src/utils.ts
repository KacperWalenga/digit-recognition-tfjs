import sharp from 'sharp';
import * as tf from '@tensorflow/tfjs-node';
import { createCanvas, loadImage } from 'canvas';
import fs from 'fs';


export async function loadImageAsT4D(imagePath: string, size = [28, 28]) {
    const imageBuffer = await sharp(imagePath)
        .resize(size[0], size[1])
        .greyscale()
        .raw()
        .toBuffer();

    const imageTensor = tf.tensor3d(new Uint8Array(imageBuffer), [28, 28, 1]);
    const normalized = imageTensor.div(255);

    return normalized.expandDims(0) as tf.Tensor4D;
}

export async function processImageFromPath(imagePath: string): Promise<tf.Tensor2D> {
  const img = await loadImage(imagePath);
  const canvas = createCanvas(200, 200);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, 200, 200);
  const imageData = ctx.getImageData(0, 0, 200, 200);

  return await convertImage(imageData as unknown as ImageData);
}

export async function processImageTensor(imgTensor: tf.Tensor3D): Promise<tf.Tensor2D> {
  const gray = tf.tidy(() => {
    const [r, g, b] = tf.split(imgTensor, 3, 2);
    const gray = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));
    return tf.sub(tf.scalar(255), gray.squeeze());
  });

  return await convertImageFromTensor(gray as tf.Tensor2D);
}

export async function saveTensorAsImage(tensor: tf.Tensor2D, path: string): Promise<void> {
  const canvas = createCanvas(28, 28);
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(28, 28);

  const data = await tensor.toInt().data();

  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    const j = i * 4;
    imageData.data[j] = val;       // R
    imageData.data[j + 1] = val;   // G
    imageData.data[j + 2] = val;   // B
    imageData.data[j + 3] = 255;   // A
  }

  ctx.putImageData(imageData, 0, 0);

  const out = fs.createWriteStream(path);
  const stream = canvas.createPNGStream();
  stream.pipe(out);

  return new Promise((resolve) => {
    out.on('finish', resolve);
  });
}

async function convertImage(imageData: ImageData): Promise<tf.Tensor2D> {
  const grayscale = tf.tidy(() => {
    const data = imageData.data;
    const gray = [];
    for (let i = 0; i < data.length; i += 4) {
      gray.push(255 - data[i]);
    }
    return tf.tensor2d(gray, [imageData.height, imageData.width], 'float32');
  });

  return await convertImageFromTensor(grayscale);
}

async function convertImageFromTensor(img: tf.Tensor2D): Promise<tf.Tensor2D> {
  let result = binarize(img);
  result = await cropEmptyBorders(result);
  result = resizeAndPadTo28x28(result);
  result = await centerImage(result);
  return result;
}

function binarize(img: tf.Tensor2D): tf.Tensor2D {
  const threshold = tf.scalar(128);
  return img.greater(threshold).mul(255).toInt() as tf.Tensor2D;
}

async function cropEmptyBorders(img: tf.Tensor2D): Promise<tf.Tensor2D> {
  const arr = await img.array();
  let top = 0, bottom = arr.length, left = 0, right = arr[0].length;

  while (top < bottom && arr[top].every(v => v === 0)) top++;
  while (bottom > top && arr[bottom - 1].every(v => v === 0)) bottom--;
  while (left < right && arr.every(row => row[left] === 0)) left++;
  while (right > left && arr.every(row => row[right - 1] === 0)) right--;

  return img.slice([top, left], [bottom - top, right - left]);
}

function resizeAndPadTo28x28(img: tf.Tensor2D): tf.Tensor2D {
  const [rows, cols] = img.shape;
  let newRows = rows, newCols = cols;
  const target = 20;

  if (rows > cols) {
    const scale = target / rows;
    newRows = target;
    newCols = Math.round(cols * scale);
  } else {
    const scale = target / cols;
    newCols = target;
    newRows = Math.round(rows * scale);
  }

  const img3d = img.expandDims(-1) as tf.Tensor3D;
  const resized = tf.image.resizeBilinear(img3d, [newRows, newCols]).squeeze() as tf.Tensor2D;

  const padTop = Math.ceil((28 - newRows) / 2);
  const padBottom = Math.floor((28 - newRows) / 2);
  const padLeft = Math.ceil((28 - newCols) / 2);
  const padRight = Math.floor((28 - newCols) / 2);

  return tf.pad(resized, [[padTop, padBottom], [padLeft, padRight]]) as tf.Tensor2D;
}

function shiftImage(img: tf.Tensor2D, shiftX: number, shiftY: number): tf.Tensor2D {
  const [h, w] = img.shape;

  const padded = tf.pad(img, [
    [Math.max(shiftY, 0), Math.max(-shiftY, 0)],
    [Math.max(shiftX, 0), Math.max(-shiftX, 0)]
  ]);

  return padded.slice(
    [Math.max(-shiftY, 0), Math.max(-shiftX, 0)],
    [h, w]
  ) as tf.Tensor2D;
}

async function centerImage(img: tf.Tensor2D): Promise<tf.Tensor2D> {
  const arr = await img.array();
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
}