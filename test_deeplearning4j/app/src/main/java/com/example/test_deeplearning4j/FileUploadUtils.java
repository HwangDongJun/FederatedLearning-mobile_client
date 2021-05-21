package com.example.test_deeplearning4j;

import android.util.Log;

import java.io.File;
import java.io.IOException;

import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;

public class FileUploadUtils {
    private static String responseString;

    public static void goSend(File file, String upload_url, OkHttpClient client) {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("files", file.getName(), RequestBody.create(MultipartBody.FORM, file))
                .build();

        Request request = new Request.Builder()
                .url(upload_url)
                .post(requestBody)
                .build();

        Response response = null;
        try {
            response = client.newCall(request).execute();
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (response.isSuccessful()) {
            ResponseBody body = response.body();
            if (body != null) {
                try {
                    responseString = body.string();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else
                Log.d("INFO", "Connect Error Occurred");

            if (!responseString.equals("file_success")) {
                Log.d("WARNING", "file not upload to server!!");
            }
        }

        response.body().close();
    }
}
