import sharp from 'sharp';
import * as tf from '@tensorflow/tfjs-node';


export async function loadImage(imagePath: string, size = [28, 28]) {
    const imageBuffer = await sharp(imagePath)
        .resize(size[0], size[1])
        .greyscale()
        .raw()
        .toBuffer();

    const imageTensor = tf.tensor(new Uint8Array(imageBuffer), size);
    return imageTensor.flatten().div(255);
}