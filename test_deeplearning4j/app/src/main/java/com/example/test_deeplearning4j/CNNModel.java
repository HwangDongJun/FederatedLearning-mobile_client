package com.example.test_deeplearning4j;

import android.util.Log;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.simple.JSONObject;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import okhttp3.OkHttpClient;

public class CNNModel implements FederatedModel {
    private static final String TAG = "CNNModel";
    private static final int BATCH_SIZE = 64;
    private static final int N_EPOCHS = 1;
    private static final int rngSeed = 42;

    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int OUTPUT_NUM = 5;

    private MultiLayerNetwork model;

    private static Logger log = LoggerFactory.getLogger(CNNModel.class);

    private String train_data_path = "/storage/self/primary/Download/data_balance/client1_train/";
    private String test_data_path = "/storage/self/primary/Download/data_balance/test/";

    private DataSetIterator AcitivityTrain;
    private DataSetIterator AcitivityTest;

    public CNNModel(int N_SAMPLES_CLIENT_TRAINING, int N_SAMPLE_CLIENT_TEST) throws IOException {
        AcitivityTrain = getDataSetIterator(train_data_path, N_SAMPLES_CLIENT_TRAINING);
        AcitivityTest = getTestDataSetIterator(test_data_path, N_SAMPLE_CLIENT_TEST);
    }

    @Override
    public void buildModel(String modelsip_path) {
        //Load the model
        try {
            File modelzip = new File(modelsip_path + "/MyMultiLayerNetwork.zip");
            model = ModelSerializer.restoreMultiLayerNetwork(modelzip);
            MultiLayerConfiguration neural_config2 = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(10)
                            .nOut(5)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .build();
            MultiLayerNetwork model2 = new MultiLayerNetwork(neural_config2);
            model2.init();

            INDArray para1_W = model.getOutputLayer().getParam("W");
            INDArray para1_b = model.getOutputLayer().getParam("b");

            model2.getLayer(0).setParam("W", para1_W);
            model2.getLayer(0).setParam("b", para1_b);

            Layer[] layers = new Layer[model.getnLayers()];
            for(int i = 0; i < model.getnLayers() - 1; i++) {
                layers[i] = model.getLayer(i);
            }
            layers[layers.length-1] = model2.getLayer(0);
            model.setLayers(layers);
            model.init();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void train(int numEpochs) throws InterruptedException {
        Log.d(TAG, " start fit!");
        model.fit(AcitivityTrain, numEpochs);
    }

    @Override
    public String eval() {
        Evaluation model_eval = model.evaluate(AcitivityTest);
        return Double.toString(model_eval.accuracy()) + "," + Double.toString(model_eval.f1());
    }

    @Override
    public void saveModel(String modelName) {
        try {
            File save_model = new File(modelName);
            model.save(save_model);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveSerializeModel(String modelName) {
        try {

            int layer_length = model.getnLayers();
            JSONObject para_json = new JSONObject();
            for(int i = 0; i < layer_length; i++) {
                if(model.getLayer(i).getParam("W") != null) {
                    // 1. W param
                    JSONArray data_W = new JSONArray();
                    INDArray param_w = model.getLayer(i).getParam("W");
                    long[] param_shape_w = param_w.shape();

                    int total_size = 1;
                    for(int j = 0; j < param_shape_w.length; j++) {
                        total_size *= param_shape_w[j];
                    }
                    INDArray reshape_param = param_w.reshape(1, total_size);
                    for (int k = 0; k < reshape_param.getRow(0).length(); k++) {
                        data_W.put(reshape_param.getRow(0).getFloat(k));
                    }

                    // 2. b param
                    JSONArray data_b = new JSONArray();
                    INDArray param_b = model.getLayer(i).getParam("b");

                    for (int k = 0; k < param_b.columns(); k++) {
                        data_b.put(param_b.getRow(0).getFloat(k));
                    }

                    para_json.put(Integer.toString(i) + "_W", data_W);
                    para_json.put(Integer.toString(i) + "_b", data_b);
                }
            }

            FileWriter file = new FileWriter("/storage/self/primary/Download/save_weight/" + modelName);
            file.write(para_json.toJSONString());
            file.flush();
            file.close();
        } catch (IOException | JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void uploadTo(String upload_path, String upload_url, OkHttpClient client) throws IOException {
        File tempSelectFile = new File(upload_path);
        FileUploadUtils.goSend(tempSelectFile, upload_url, client);
    }

    @Override
    public DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
        File train_data = new File(folderPath);
        FileSplit train = new FileSplit(train_data, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(112, 112, 3, labelMaker);

        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 16, 1, 5);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        return dataIter;
    }

    private static DataSetIterator getTestDataSetIterator(String folderPath, int nSamples) throws IOException {
        File folder = new File(folderPath);
        File[] digitFolders = folder.listFiles();

        NativeImageLoader nil = new NativeImageLoader(112, 112, 3);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);

        INDArray input = Nd4j.create(new int[]{ nSamples, 3, 112, 112 });
        INDArray output = Nd4j.create(new int[]{ nSamples, 5 });

        int n = 0;
        //scan all 0..9 digit subfolders
        for (File digitFolder : digitFolders) {
            //take note of the digit in processing, since it will be used as a label
            int labelDigit = 0;
            String file_digit = digitFolder.getName();
            if(file_digit.equals("book")) {
                labelDigit = 0;
            } else if(file_digit.equals("laptop")) {
                labelDigit = 1;
            } else if(file_digit.equals("phone")) {
                labelDigit = 2;
            } else if(file_digit.equals("wash")) {
                labelDigit = 3;
            } else if(file_digit.equals("water")) {
                labelDigit = 4;
            }
            //scan all the images of the digit in processing
            File[] imageFiles = digitFolder.listFiles();
            for (File imageFile : imageFiles) {
                //read the image as a one dimensional array of 0..255 values
//				INDArray img = nil.asRowVector(imageFile);
                INDArray img = nil.asMatrix(imageFile);
                log.info(img.shapeInfoToString());
                //scale the 0..255 integer values into a 0..1 floating range
                //Note that the transform() method returns void, since it updates its input array
                scaler.transform(img);
                //copy the img array into the input matrix, in the next row
                input.putRow( n, img );
                //in the same row of the output matrix, fire (set to 1 value) the column correspondent to the label
                output.put( n, labelDigit, 1.0 );
                //row counter increment
                n++;
                log.info(labelDigit+" \t "+n);
            }
        }

        //Join input and output matrixes into a dataset
        DataSet dataSet = new DataSet( input, output );
        //Convert the dataset into a list
        List<DataSet> listDataSet = dataSet.asList();
        //Shuffle its content randomly
        Collections.shuffle( listDataSet, new Random(System.currentTimeMillis()) );
        //Set a batch size
        int batchSize = 32;
        //Build and return a dataset iterator that the network can use
        DataSetIterator dsi = new ListDataSetIterator<DataSet>( listDataSet, batchSize );
        return dsi;
    }
}
