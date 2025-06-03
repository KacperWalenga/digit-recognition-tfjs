import * as tf from '@tensorflow/tfjs-node';
import mnist from 'mnist';

export class MnistLoader {
  trainImages!: tf.Tensor4D;
  trainLabels!: tf.Tensor2D;
  testImages!: tf.Tensor4D;
  testLabels!: tf.Tensor2D;

  constructor(private trainSize = 60000, private testSize = 10000) {}

  loadData() {
    const set = (mnist as any).set(this.trainSize, this.testSize);
    const trainingSet = set.training;
    const testSet = set.test;

    const trainImagesArray = trainingSet.map((x: any) => x.input);
    const trainLabelsArray = trainingSet.map((x: any) => x.output);

    const testImagesArray = testSet.map((x: any) => x.input);
    const testLabelsArray = testSet.map((x: any) => x.output);

    this.trainImages = tf.tensor2d(trainImagesArray)
      .reshape([trainImagesArray.length, 28, 28, 1]) as tf.Tensor4D;

    this.trainLabels = tf.tensor2d(trainLabelsArray);

    this.testImages = tf.tensor2d(testImagesArray)
      .reshape([testImagesArray.length, 28, 28, 1]) as tf.Tensor4D;

    this.testLabels = tf.tensor2d(testLabelsArray);
  }
}
