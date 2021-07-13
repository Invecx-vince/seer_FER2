package com.example.imagepro;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.media.AudioManager;
import android.media.MediaActionSound;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.Window;
import android.view.WindowManager;
import android.speech.tts.TextToSpeech;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="MainActivity";

    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private ImageView flip;
    private TextToSpeech speaker;
    private ImageView takeImg;
    private TextView emoTxt;
    private fer face_recognizer;
    int counter = 0;
    private int cameraState = 0;
    private int img = 0;
    // 0 = backcam
    // 1 = frontcam
    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public CameraActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE}, MY_PERMISSIONS_REQUEST_CAMERA);
        }
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.READ_EXTERNAL_STORAGE}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_camera);

        speaker = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int i) {

                // if No error is found then only it will run
                if(i!=TextToSpeech.ERROR){
                    // To Choose language of speech
                    speaker.setLanguage(Locale.UK);
                }
            }
        });

        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(1280, 760);
        flip = (ImageView)findViewById(R.id.button);
        emoTxt = (TextView)findViewById(R.id.emotionView);
        takeImg = (ImageView)findViewById(R.id.take_pic_btn);
        flip.setOnClickListener(new OnClickListener(){
            @Override
            public void onClick(View view) {
                swapCam();
            }
        });
        takeImg.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                if(img==0)
                    img=1;
                else
                    img=0;
            }
        });

/*
        mButton = (Button)findViewById(R.id.button);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {//for taking a picture
                MediaActionSound sfx = new MediaActionSound();
                sfx.play(MediaActionSound.SHUTTER_CLICK);


            }
        });*/


        try{
            int inputSize = 48;
            face_recognizer = new fer(getAssets(),CameraActivity.this,
                    "model300.tflite",inputSize);
                    // "neo_model.tflite",inputSize);
        }
        catch(IOException e){
            e.printStackTrace();
        }
    }

    private void swapCam() {
        cameraState = cameraState^1;
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(cameraState);
        mOpenCvCameraView.enableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            //if load success
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if not loaded
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);
    }
    public void onCameraViewStopped(){
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        mRgba=inputFrame.rgba();
        mGray=inputFrame.gray();
        if(cameraState==1){
            Core.flip(mRgba,mRgba,-1);
            Core.flip(mGray,mGray,-1);
        }

        //img = img_capture(img, mRgba);
        if(img==1){
            String test = face_recognizer.readImage2(mRgba);
            HashMap<String, String> myHashAlarm = new HashMap<String, String>();
            myHashAlarm.put(TextToSpeech.Engine.KEY_PARAM_STREAM, String.valueOf(AudioManager.STREAM_ALARM));
            myHashAlarm.put(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "yeeeet");
            speaker.speak(test, TextToSpeech.QUEUE_FLUSH, myHashAlarm);
            img=0;
        }
        mRgba = face_recognizer.readImage(mRgba);
        return mRgba;

    }

    private int img_capture(int takeImg,Mat mRgba) {
        if(takeImg==1){
            Mat save = new Mat();
            Core.flip(mRgba.t(), save, 1);
            Imgproc.cvtColor(save, save, Imgproc.COLOR_mRGBA2RGBA);
            String test = face_recognizer.readImage2(mRgba);
            String speech = "Emotions detected are: " + test;
            emoTxt.setText("Emotion: " + test);
            HashMap<String, String> myHashAlarm = new HashMap<String, String>();
            myHashAlarm.put(TextToSpeech.Engine.KEY_PARAM_STREAM, String.valueOf(AudioManager.STREAM_ALARM));
            myHashAlarm.put(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "yeeeet");
            speaker.speak(speech, TextToSpeech.QUEUE_FLUSH, myHashAlarm);
        }
        return 0;
    }


}