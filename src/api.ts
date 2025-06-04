import express from 'express';
import multer from 'multer';
import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import fs from 'fs';
import { engine } from 'express-handlebars';
import { loadModel } from './model';
import { processImageFromPath } from './utils';

const app = express();
const port = process.env.PORT || 3000;

app.engine('handlebars', engine({
  defaultLayout: 'main',
  layoutsDir: path.join(__dirname, '../../views/layouts'),
  partialsDir: path.join(__dirname, '../../views/partials')
}));
app.set('view engine', 'handlebars');
app.set('views', path.join(__dirname, '../../views'));

app.use('/static', express.static(path.join(__dirname, '../../public')));
app.use('/uploads', express.static('uploads'));

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const uploadDir = './uploads';
      if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir, { recursive: true });
      }
      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
      cb(null, uniqueSuffix + path.extname(file.originalname));
    }
  }),
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Nieprawidłowy format pliku. Akceptowane są tylko obrazy.'));
    }
  },
  limits: {
    fileSize: 5 * 1024 * 1024
  }
});

let model: tf.LayersModel;

app.post('/recognize', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Brak pliku obrazu' });
    }

    const imagePath = req.file.path;
    const imageFilename = req.file.filename;
    
    const processed = await processImageFromPath(imagePath);
    const input = processed
      .toFloat()
      .div(255)
      .expandDims(0)
      .expandDims(-1);

    const predictions = model.predict(input) as tf.Tensor;
    const predictionArray = await predictions.array() as number[][];
    
    const allProbabilities = predictionArray[0].map((prob, index) => ({
      digit: index,
      probability: prob,
      percentage: (prob * 100).toFixed(2)
    }));
    
    allProbabilities.sort((a, b) => b.probability - a.probability);
    
    const predictedClass = allProbabilities[0].digit;
    const probability = allProbabilities[0].probability;

    tf.dispose([processed, input, predictions]);
    
    const wantsJson = req.headers.accept?.includes('application/json');
    
    if (wantsJson) {
      fs.unlinkSync(imagePath);
      
      return res.json({
        predictedDigit: predictedClass,
        probability: probability,
        confidencePercentage: (probability * 100).toFixed(2) + '%',
        allProbabilities: allProbabilities
      });
    }
    
    res.render('result', {
      title: `Rozpoznano cyfrę: ${predictedClass}`,
      predictedDigit: predictedClass,
      confidence: (probability * 100).toFixed(2),
      imageFilename: imageFilename,
      allProbabilities: allProbabilities
    });

    setTimeout(() => {
      try {
        if (fs.existsSync(imagePath)) {
          fs.unlinkSync(imagePath);
        }
      } catch (e) {
        console.error('Błąd podczas usuwania pliku:', e);
      }
    }, 5 * 60 * 1000);

  } catch (error) {
    console.error('Błąd podczas rozpoznawania:', error);
    res.status(500).json({ error: 'Wystąpił błąd podczas przetwarzania obrazu' });
  }
});

app.get('/recognize-api', (req, res) => {
  res.render('api-docs', {
    title: 'Dokumentacja API Rozpoznawania Cyfr',
    port: port
  });
});

app.get('/', (req, res) => {
  res.render('home', {
    title: 'API Rozpoznawania Cyfr'
  });
});

app.get('/recognize', (req, res) => {
  res.redirect('/');
});

async function startServer() {
  try {
    console.log('Ładowanie modelu...');
    model = await loadModel();
    console.log('Model załadowany pomyślnie!');
    
    app.listen(port, () => {
      console.log(`Serwer działa na porcie: ${port}`);
    });
  } catch (error) {
    console.error('Błąd podczas ładowania modelu:', error);
    process.exit(1);
  }
}

startServer();