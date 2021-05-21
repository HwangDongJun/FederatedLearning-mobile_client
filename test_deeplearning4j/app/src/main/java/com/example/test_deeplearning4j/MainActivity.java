package com.example.test_deeplearning4j;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;

public class MainActivity extends AppCompatActivity {
    private EditText clientAddress;
    private TextView stepText;
    private TextView logArea;
    private static final String TAG = "FederatedActivity";
    private String mUsername;
    Handler handler = null;
    private CNNModel cnn_model;
    private int model_version;
    private String temp_modelstate;
    private OkHttpClient client;
    private int currentRound = 0;
    private int max_train_round = 0;
    private Response response = null;
    private String responseString;
    private String return_task;
    private JSONObject response_json = null;
    private String address = "192.168.0.100";
    private double ui_testacc = 0.0;

    List<Entry> entryList = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        MultiDex.install(this);

        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, MODE_PRIVATE);

        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");

//        mChart = findViewById(R.id.chart);
        clientAddress = (EditText) findViewById(R.id.ClientAddress);
        stepText = findViewById(R.id.step);
        logArea = findViewById(R.id.log_area);

        Button startManualBtn = findViewById(R.id.btn_start_manual);
        Button preProcessBtn = findViewById(R.id.btn_preprocess);
        entryList.add(new Entry(0, 0));
        LineDataSet lineDataSet = new LineDataSet(entryList, "loss");
        LineData lineData = new LineData(lineDataSet);

        handler = new Handler(msg -> {
            Bundle bundle = msg.getData();
            String str = bundle.getString("data");
            logArea.append(str + "\n");
            return false;
        });

        preProcessBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(clientAddress.getText().length() != 0) {
                    address = clientAddress.getText().toString();
                }

                mUsername = "android_" + Math.floor((Math.random() * 1000) + 1);
                client = new OkHttpClient.Builder()
                        .connectTimeout(6000, TimeUnit.MINUTES)
                        .writeTimeout(6000, TimeUnit.MINUTES)
                        .readTimeout(6000, TimeUnit.MINUTES)
                        .build();

                Message msg = new Message();
                Bundle bundle = new Bundle();
                bundle.putString("data", "create android user");
                msg.setData(bundle);
                handler.sendMessage(msg);
            }
        });

        startManualBtn.setOnClickListener(v -> {
            try {
                // 예외상황 처리해야함
                // 1. 연결 끊겼을 경우 disconnect기능 구현해서 서버쪽에도 알려줘야함
                // 2.

                // client wake up!
                wakeup();

                // client ready!
                oninit();

                Message msg = new Message();
                Bundle bundle = new Bundle();
                bundle.putString("data", "success init");
                msg.setData(bundle);
                handler.sendMessage(msg);
                Log.d("INFO", "success init");

                // download model (server -> client)
                download();
                // First Round Training
                response_json = new JSONObject(responseString);
                max_train_round = response_json.getInt("max_train_round");
                update();

                response_json = new JSONObject(responseString);
                while(response_json.getInt("current_round") <= max_train_round) {
                    // 서버와 클라이언트 라운드 학습 스케쥴링
                    if (response_json.getString("state").equals("RESP_ARY")) { // 1 라운드의 모든 클라이언트 학습은 끝났지만 모든 라운드 학습은 하지 못함
                        download();
                        update();
                    } else if (response_json.getString("state").equals("RESP_ACY")) { // 1 라운드에서 아직 다른 클라이언트가 학습을 완료하지 못함
                        do {
                            Thread.sleep(60000);
                            version();
                            response_json = new JSONObject(responseString);
                        } while (temp_modelstate.equals("WAIT"));

                        if(response_json.getInt("current_round") > max_train_round) {
                            break;
                        }

                        download();
                        update();
                    } else if (response_json.getString("state").equals("NEC")) {
//                        Message msg = new Message();
//                        Bundle bundle = new Bundle();
//                        bundle.putString("data", "Not equal currentRound error");
//                        msg.setData(bundle);
//                        handler.sendMessage(msg);
                        Log.d("WARNING", "Not equal currentRound error");
                    }
                    response_json = new JSONObject(responseString);
                    model_version = response_json.getInt("model_version");
                }

//                Message msg = new Message();
//                Bundle bundle = new Bundle();
//                bundle.putString("data", "Final accuracy " + Double.toString(response_json.getDouble("model_acc")));
//                msg.setData(bundle);
//                handler.sendMessage(msg);
                Log.d("RESULT INFO", "Final accuracy " + Double.toString(response_json.getDouble("model_acc")));

                if(response_json.getString("state").equals("FIN")) {
//                    msg = new Message();
//                    bundle = new Bundle();
//                    bundle.putString("data", "Final accuracy " + Double.toString(response_json.getDouble("model_acc")));
//                    msg.setData(bundle);
//                    handler.sendMessage(msg);
                    Log.d("RESULT INFO", "Final accuracy " + Double.toString(response_json.getDouble("model_acc")));
                }

            } catch (JSONException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        });
    }

    private void wakeup() throws ExecutionException, InterruptedException {
        return_task = new WakeupClientTask().execute().get();
    }

    private void oninit() throws ExecutionException, InterruptedException {
        AsyncTask<Void, Integer, String> init_task = new OninitTask().execute();
        return_task = init_task.get();
        init_task.cancel(true);
        if(return_task == "NOT REGISTER CLIENT") {
//            Message msg = new Message();
//            Bundle bundle = new Bundle();
//            bundle.putString("data", "not register client");
//            msg.setData(bundle);
//            handler.sendMessage(msg);
            Log.d("INFO", "Not register client");

            // 예외상황 처리해야함
        }
    }

    private void update() throws ExecutionException, InterruptedException {
        AsyncTask<Void, Integer, String> update_task = new RequestUpdateTask().execute();
        return_task = update_task.get();
        update_task.cancel(true);
    }

    private void download() {
        AsyncTask<Void, String, String> download_task = new DownloadFileFromURL().execute();
        download_task.cancel(true);
    }

    private void version() throws ExecutionException, InterruptedException {
        AsyncTask<Void, Integer, String> version_task = new CheckModelVersion().execute();
        return_task = version_task.get();
        version_task.cancel(true);
    }

    class DownloadFileFromURL extends AsyncTask<Void, String, String> {
        @Override
        protected String doInBackground(Void... voids) {
            int count;
            try {
//                URL url = new URL("http://" + address + ":8890/saved_model/MyMultiLayerNetwork.zip");
                URL url = new URL("http://" + address + ":8891/download");
                URLConnection conection = url.openConnection();
                conection.connect();

                // download the file
                InputStream input = new BufferedInputStream(url.openStream(),
                        10240);

                // Output stream
                OutputStream output = new FileOutputStream("/storage/self/primary/Download/save_model/MyMultiLayerNetwork.zip");

                byte data[] = new byte[10240];

//                long total = 0;

                while ((count = input.read(data)) != -1) {
                    // writing data to file
                    output.write(data, 0, count);
                }

                // flushing output
                output.flush();

                // closing streams
                output.close();
                input.close();
            } catch (Exception e) {
                Log.e("Error: ", e.getMessage());
            }

            return null;
        }
    }

    class WakeupClientTask extends AsyncTask<Void, Integer, String> {
        @Override
        protected String doInBackground(Void... voids) {
            // client wake up!
            Request request = new Request.Builder()
                    .url("http://" + address + ":8891/client_wake_up?client_name=" + mUsername)
                    .build();

            try {
                response = client.newCall(request).execute();
                if (response.isSuccessful()) {
                    ResponseBody body = response.body();
                    if (body != null) {
                        responseString = body.string();
                    }
                }
                else
                    Log.d("INFO", "Connect Error Occurred");
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                response.body().close();
            }

            return responseString;
        }

        @Override
        protected void onPostExecute(String result) {
            logArea.append("send wake up \n");
            Log.d("INFO", "send wake up");
            Log.d("INFO", "Response from the server : " + responseString);
        }
    }

    class OninitTask extends AsyncTask<Void, Integer, String> {
        @Override
        protected String doInBackground(Void... voids) {
            // client ready!
            String Output_file_path = "/storage/self/primary/Download/save_model";
            File check_file = new File(Output_file_path);

            if(!check_file.exists()) {
                boolean success = check_file.mkdir();
            }

            try {
                cnn_model = new CNNModel();

                Integer trainSize = 468;

                Request request2 = new Request.Builder()
                        .url("http://" + address + ":8891/client_ready?client_name=" + mUsername + "&train_size=" + Integer.toString(trainSize) + "&model_ver=" + model_version + "&current_round=" + Integer.toString(currentRound))
                        .build();

                response = client.newCall(request2).execute();
                if (response.isSuccessful()) {
                    ResponseBody body = response.body();
                    if (body != null) {
                        responseString = body.string();
                    }
                }
                else
                    Log.d("INFO", "Connect Error Occurred");
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                response.body().close();
            }

            return responseString;
        }

        @Override
        protected void onPostExecute(String result) {
//            logArea.append("client ready \n");
//            Message msg = new Message();
//            Bundle bundle = new Bundle();
//            bundle.putString("data", "success init");
//            msg.setData(bundle);
//            handler.sendMessage(msg);
            Log.d("INFO", "success init");
        }
    }

    class RequestUpdateTask extends AsyncTask<Void, Integer, String> {
        @Override
        protected String doInBackground(Void... voids) {
            String Train_time = "";
            // requeest update
            try {
                int mVersion = response_json.getInt("model_version");
                currentRound = response_json.getInt("current_round");
                double testLoss = response_json.getDouble("model_loss");
                double testAcc = response_json.getDouble("model_acc");
                String upload_url = response_json.getString("upload_url");
                String download_url = response_json.getString("model_url");
                String Output_file_path = "/storage/self/primary/Download/save_model";
                File check_file = new File(Output_file_path);

                if(mVersion != model_version || mVersion == 0) {
                    // model build
                    cnn_model.buildModel(Output_file_path);
                }

                ui_testacc = testAcc;

                Train_time = trainOneRound(currentRound, upload_url, mVersion);

                Request update_client_request = new Request.Builder()
                        .url("http://" + address + ":8891/client_update?client_name=" + mUsername + "&current_round=" + Integer.toString(currentRound))
                        .build();

                response = client.newCall(update_client_request).execute();

                if (response.isSuccessful()) {
                    ResponseBody body = response.body();
                    if (body != null) {
                        responseString = body.string();
                    }
                }
                else
                    Log.d("INFO", "Connect Error Occurred");
            } catch (JSONException | IOException e) {
                e.printStackTrace();
            } finally {
                response.body().close();
            }
            return Train_time;
        }

        @Override
        protected void onPostExecute(String result) {
//            String acc_msg = "global acc: " + ui_testacc + "\n";
//            logArea.append(acc_msg);
//            stepText.setText(getString(R.string.current_round, currentRound));
//
//            Message msg = new Message();
//            Bundle bundle = new Bundle();
//            bundle.putString("data", result);
//            msg.setData(bundle);
//            handler.sendMessage(msg);
            Log.d("TRAINING TIME INFO", result);
        }
    }

    class CheckModelVersion extends AsyncTask<Void, Integer, String> {
        @Override
        protected String doInBackground(Void... voids) {
            Request update_client_request = new Request.Builder()
                    .url("http://" + address + ":8891/model_version?version_client=" + mUsername + "&model_ver=" + model_version)
                    .build();

            try {
                response = client.newCall(update_client_request).execute();

                if (response.isSuccessful()) {
                    ResponseBody body = response.body();
                    if (body != null) {
                        try {
                            responseString = body.string();
                            temp_modelstate = new JSONObject(responseString).getString("state");
                        } catch (IOException | JSONException e) {
                            e.printStackTrace();
                        }
                    }
                } else
                    Log.d("INFO", "Connect Error Occurred");
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                response.body().close();
            }

            return temp_modelstate;
        }

        @Override
        protected void onPostExecute(String result) {
            String modelVersion = "curent model version: " + result;
//            logArea.append(modelVersion);
//
//            Message msg = new Message();
//            Bundle bundle = new Bundle();
//            bundle.putString("data", modelVersion);
//            msg.setData(bundle);
//            handler.sendMessage(msg);
            Log.d("INFO", modelVersion);
        }
    }

    private String trainOneRound(int currentRound, String upload_url, int modelVersion) throws IOException {
        Log.d(TAG, "execute: train start!");
        long current_time = 0L;
        long train_time = 0L;
        try {
            current_time = System.currentTimeMillis();
            cnn_model.train(1);
            train_time = System.currentTimeMillis();
            Log.d("TRAINING TIME INFO", Long.toString((train_time - current_time)/1000));
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Log.d(TAG, "run: train finish!");

        // save trained model
        String AndroidModelPath = "/storage/self/primary/Download/save_weight/";

        cnn_model.saveSerializeModel("weight_" + mUsername + ".json");

//        Message msg = new Message();
//        Bundle bundle = new Bundle();
//        bundle.putString("data", "Complete model save");
//        msg.setData(bundle);
//        handler.sendMessage(msg);
        Log.d("MODEL INFO", "Complete model save!");
        // upload to server trained model
        cnn_model.uploadTo(AndroidModelPath + "weight_" + mUsername + ".json", upload_url, client);

        return Long.toString((train_time - current_time)/1000);
    }
}