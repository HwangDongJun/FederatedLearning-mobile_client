package com.example.test_deeplearning4j;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

import okhttp3.OkHttpClient;

public interface FederatedModel {

    void buildModel(String modelsip_path);

    // void train(TrainerDataSource trainerDataSource);
    void train(int numEpochs) throws InterruptedException;

    void saveModel(String modelName);

    void saveSerializeModel(String modelName);

    void uploadTo(String upload_path, String upload_url, OkHttpClient client) throws IOException;

    DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException;

}
