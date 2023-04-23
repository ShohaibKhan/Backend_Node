
const express = require('express');
const cors = require('cors')
const fs = require('fs');
const bodyparser = require('body-parser')
const multer = require('multer');
const path = require('path')
const app = express();
const tf = require("@tensorflow/tfjs");
const wav = require('wav-decoder');

let predictedOutput = null;


app.use(bodyparser.urlencoded({extended:false}))
app.use(bodyparser.json())

const buffer = fs.readFileSync('model.h5');


const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now()+path.extname(file.originalname));
  },
});

const modelConfig = {"class_name": "Sequential", "config": {"name": "mfcc", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 40, 862, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "batch_input_shape": [null, 40, 862, 1], "dtype": "float32", "filters": 16, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last", "keepdims": false}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.9.0", "backend": "tensorflow"}

async function loadModel() {
    console.log("inside load")
    console.log(modelConfig)
    const model = await tf.models.modelFromJSON(modelConfig);
    
    return model;
}
async function preprocessAudioData(audioData) {
  // Normalize the audio waveform
  const normalizedAudioData = audioData.channelData[0].map(sample => sample / Math.abs(sample)).slice(0, 16000);

  // Pad the audio waveform if necessary
  const paddedAudioData = normalizedAudioData.length < 34480 ? [...normalizedAudioData, ...Array(34480 - normalizedAudioData.length).fill(0)] : normalizedAudioData;

  // Reshape the audio waveform
  const reshapedAudioData = tf.tensor2d(paddedAudioData, [1, 34480]);

  return reshapedAudioData.reshape([-1, 40, 862, 1]);
}
const upload = multer({storage: storage,}).single('file');

app.use(express.static('uploads'))
app.use(cors())

app.post("/predict",(req,res)=>{
    upload(req,res,async (err)=>{
        if (err){
            console.log(err)
        }
        console.log(req.file);
        const arrayBuffer = fs.readFileSync(req.file.path);
        const audioData = await wav.decode(arrayBuffer);

      // preprocess the audio data
        inputData = await preprocessAudioData(audioData);
        
        
        
        console.log("th inout data is",inputData)

        const model = await loadModel();
        console.log(model.summary());
        const prediction = await model.predict(inputData);
        predictedOutput = prediction.dataSync();
        console.log("data sync is",  prediction.dataSync());
        console.log("array sync is", prediction.arraySync())
        console.log("prediction is",prediction);
        console.log("iam inside the predict api",model.summary);
        
        
    })
})

app.get('/display', (req, res) => {
  // Do some processing to get the prediction result
  const predictionResult = [1, 2, 3, 4, 5];

  // Send the prediction result as the response data
  res.send(predictedOutput);
  predictedOutput = null;
});
app.get("/",(req,res)=>{
    res.send("Iam express Server Again")
})


app.listen(3000, () => {
  console.log('Server listening on port 3000');
});



