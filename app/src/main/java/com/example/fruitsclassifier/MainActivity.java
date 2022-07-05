package com.example.fruitsclassifier;

import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.fruitsclassifier.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button camera,gallery;
    ImageView imageView;
    TextView result;
    int imageSize=32;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera=findViewById(R.id.button);
        gallery=findViewById(R.id.button2);
        result=findViewById(R.id.result);
        imageView=findViewById(R.id.imageView);

//        asking camera access
       camera.setOnClickListener(new View.OnClickListener() {
           @RequiresApi(api = Build.VERSION_CODES.M)
           @Override
           public void onClick(View view) {
               if(ContextCompat.checkSelfPermission(MainActivity.this,Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                   Intent cameraIntent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                   startActivityForResult(cameraIntent,3);
               }else{
                  requestPermissions(new String[]{Manifest.permission.CAMERA},100);
               }
           }
       });

//       getting the image form gallery
        gallery.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                    Intent cameraIntent=new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(cameraIntent,1);
            }
        });

    }

//    ml model

    public  void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
//            allocating space for the byte buffer
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imageSize*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues= new int[imageSize*imageSize];
            image.getPixels(intValues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel = 0;
//          iterating over the image pixel values
            for(int i=0;i<imageSize;i++){
                for (int j=0;j<imageSize;j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val>>16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val>>8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF)*(1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

//            showing the output to the user
//            find the index of the class with the highest confidence
            float[] confidences =outputFeature0.getFloatArray();
            int maxPos=0;
            float maxConfidence=0;
            for (int i=0;i<confidences.length;i++){
                if(confidences[i]>maxConfidence){
                    maxConfidence=confidences[i];
                    maxPos=i;
                }
            }

            String[] classes={"Apple","Banana","Orange"};
            result.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }

//    resizing the image
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data){
        if(resultCode==RESULT_OK){
//            image captured form the camera case
            if(requestCode==3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension =Math.min(image.getWidth(),image.getHeight());
                image= ThumbnailUtils.extractThumbnail(image,dimension,dimension);
                imageView.setImageBitmap(image);

                image=Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
//                calling the model to classify the image
                classifyImage(image);
            }
            else{
//              handling the gallery selected image
                Uri dat=data.getData();
                Bitmap image=null;
                try{
                    image= MediaStore.Images.Media.getBitmap(this.getContentResolver(),dat);
                }
                catch(IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image=Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode,resultCode,data);
    }


}