package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class fer {
    private Interpreter interpreter;

    private int INPUT_SIZE;
    private int height = 0;
    private int width = 0;

    private GpuDelegate gpuDelegate = null;

    private CascadeClassifier cascadeClassifier;

    fer(AssetManager assetManager, Context context, String modelPath, int inputSize) throws IOException {
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4);
        interpreter = new Interpreter(loadModelFile(assetManager,modelPath),options);

        Log.d("fer","Model successfully loaded");
        try{
            InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = context.getDir("cascade",Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir,"haarcascade_frontalface_alt");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int byteRead;

            while ((byteRead=is.read(buffer))!=-1){
                os.write(buffer,0,byteRead);
            }
            is.close();
            os.close();
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            Log.d("fer","Classifier loaded");
        }
        catch(IOException e){
            e.printStackTrace();
        }

    }

    public Mat readImage(Mat mat_image){
        //flip 90 degrees
        Core.flip(mat_image.t(),mat_image,1);
        Mat grayImg = new Mat();
        Imgproc.cvtColor(mat_image,grayImg,Imgproc.COLOR_RGBA2GRAY);
        height = grayImg.height();
        width = grayImg.width();

        int absFaceSize = (int)(height*0.1);
        MatOfRect faces = new MatOfRect();
        if(cascadeClassifier != null){
            cascadeClassifier.detectMultiScale(grayImg,faces,1.1,2,2,
                    new Size(absFaceSize,absFaceSize),new Size());
        }
        Rect[] f_arr= faces.toArray();
        for(int i=0;i<f_arr.length;i++){
            Imgproc.rectangle(mat_image,f_arr[i].tl(),f_arr[i].br(),new Scalar(0,255,0,255),1);
            Rect crpd_face = new Rect((int)f_arr[i].tl().x,(int)f_arr[i].tl().y,
                    ((int)f_arr[i].br().x)-(int)(f_arr[i].tl().x),
                    ((int)f_arr[i].br().y)-(int)(f_arr[i].tl().y)
                    );
            Mat crpd_face_rgba = new Mat(mat_image,crpd_face);
            Bitmap bitmap = null;
            bitmap = Bitmap.createBitmap(crpd_face_rgba.cols(),crpd_face_rgba.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(crpd_face_rgba,bitmap);

            //Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,48,48,false);
            //ByteBuffer byteBuffer = convertBitmaptoByteBuffer(scaledBitmap);
            //float[][] emotion = new float[1][1];
            //interpreter.run(byteBuffer,emotion);
            //Log.d("expression", "Output "+ Array.get(Array.get(emotion,0),0));

            //float emotion_v = (float)Array.get(Array.get(emotion,0),0);
            //String emotion_s = getEmoText(emotion_v);
            //Imgproc.putText(mat_image,emotion_s+" ("+emotion_v+") ",
            //        new Point((int)f_arr[i].tl().x+10,(int)f_arr[i].tl().y+20),
            //        1,1.5,new Scalar(0,0,255,150),3);
        }

        //re-flip back to original orientation
        Core.flip(mat_image.t(),mat_image,0);
        return mat_image;
    }
    public String readImage2(Mat mat_image){
        //flip 90 degrees
        String emotion_s="";
        Core.flip(mat_image.t(),mat_image,1);
        Mat grayImg = new Mat();
        Imgproc.cvtColor(mat_image,grayImg,Imgproc.COLOR_RGBA2GRAY);
        height = grayImg.height();
        width = grayImg.width();

        int absFaceSize = (int)(height*0.1);
        MatOfRect faces = new MatOfRect();
        if(cascadeClassifier != null){
            cascadeClassifier.detectMultiScale(grayImg,faces,1.1,2,2,
                    new Size(absFaceSize,absFaceSize),new Size());
        }
        Rect[] f_arr= faces.toArray();

        for(int i=0;i<f_arr.length;i++){
            Imgproc.rectangle(mat_image,f_arr[i].tl(),f_arr[i].br(),new Scalar(0,255,0,255),1);

            Rect crpd_face = new Rect((int)f_arr[i].tl().x,(int)f_arr[i].tl().y,
                    ((int)f_arr[i].br().x)-(int)(f_arr[i].tl().x),
                    ((int)f_arr[i].br().y)-(int)(f_arr[i].tl().y)
            );
            Mat crpd_face_rgba = new Mat(mat_image,crpd_face);
            Bitmap bitmap = null;
            bitmap = Bitmap.createBitmap(crpd_face_rgba.cols(),crpd_face_rgba.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(crpd_face_rgba,bitmap);

            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,48,48,false);
            ByteBuffer byteBuffer = convertBitmaptoByteBuffer(scaledBitmap);
            float[][] emotion = new float[1][1];
            interpreter.run(byteBuffer,emotion);
            //Log.d("expression", "Output "+ Array.get(Array.get(emotion,0),0));

            float emotion_v = (float)Array.get(Array.get(emotion,0),0);
            emotion_s += getEmoText(emotion_v)+" ";
            //Imgproc.putText(mat_image,emotion_s+" ("+emotion_v+") ",
            //        new Point((int)f_arr[i].tl().x+10,(int)f_arr[i].tl().y+20),
            //        1,1.5,new Scalar(0,0,255,150),3);
        }

        //re-flip back to original orientation
        //Core.flip(mat_image.t(),mat_image,0);
        return emotion_s;
    }

    private String getEmoText(float emotion_v) {
        String val = "";
        int num = Math.round(emotion_v);
        ///*
        switch (num){
            case 0:
                val="Surprise";
                break;
            case 1:
                val="Fear";
                break;
            case 2:
                val="Angry";
                break;
            case 3:
                val="Neutral";
                break;
            case 4:
                val="Sad";
                break;
            case 5:
                val="Disgust";
                break;
            case 6:
                val="Happy";
                break;
            default:
                val = "Unknown";
        }

         //*/
/*
        if(emotion_v<1)
            val="Surprise";
        else if(emotion_v<2)
            val="Fear";
        else if(emotion_v<3)
            val="Angry";
        else if(emotion_v<4)
            val="Neutral";
        else if(emotion_v<5)
            val="Sad";
        else if(emotion_v<6)
            val="Disgust";
        else if(emotion_v<7)
            val="Happy";
*/
        return val;
    }

    private ByteBuffer convertBitmaptoByteBuffer(Bitmap scaled){
        ByteBuffer byteBuffer;
        int size_image = INPUT_SIZE;

        byteBuffer = ByteBuffer.allocateDirect(4*1*size_image*size_image*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_image*size_image];
        scaled.getPixels(intValues,0,scaled.getWidth(),0,0,scaled.getWidth(),scaled.getHeight());

        int pixel = 0;
        for(int i=0;i<size_image;i++){
            for(int j=0;j<size_image;j++){
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                byteBuffer.putFloat(((val&0xFF))/255.0f);
            }
        }

        return byteBuffer;
    }
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException{
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

}
