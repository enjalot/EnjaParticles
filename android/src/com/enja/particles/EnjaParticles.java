package com.enja.particles;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.app.Activity;
import android.content.Context;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.MotionEvent;

import android.util.Log;

public class EnjaParticles extends Activity
{
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        mGLView = new EnjGLSurfaceView(this);
        setContentView(mGLView);
        //setContentView(R.layout.main);
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        mGLView.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mGLView.onResume();
    }

    private GLSurfaceView mGLView;

    static {
        System.loadLibrary("enjaparticles");
    }
    

}

class EnjGLSurfaceView extends GLSurfaceView {
    //private boolean moving = false;
    private static final String TAG = "EnjaParticles";
    public EnjGLSurfaceView(Context context) {
        super(context);
        mRenderer = new EnjRenderer();
        setRenderer(mRenderer);
    }

    public boolean onTouchEvent(final MotionEvent event) {
        
       
        if(event.getAction() == MotionEvent.ACTION_DOWN) {
            //Log.i(TAG, "Action Down");
            int pid = event.getPointerId(0);
            float x = event.getX(pid);
            float y = event.getY(pid);
            nativeDown(x,y);
        }
        else if(event.getAction() == MotionEvent.ACTION_MOVE) {
            //moving = true;
            //Log.i(TAG, "MOVING");
            //should probably check which pointer we are dealing with
            int pid = event.getPointerId(0);
            float x = event.getX(pid);
            float y = event.getY(pid);
            nativeMove(x, y);
        }
        else if (event.getAction() == MotionEvent.ACTION_UP) {
            //nativePause();
            //if(!moving) //different behavior for just a "click" rather than "drag"
            long dt = event.getDownTime();
            long et = event.getEventTime();
            //Log.i(TAG, "dt: " + dt + "et: " + et + "et-dt: " + (et-dt));
            int pid = event.getPointerId(0);
            //Log.i(TAG, "pointerid[0]: " + pid);
            if (et-dt < 200)
            {
                //Log.i(TAG, "less than 200 ms");
                float x = event.getX(pid);
                float y = event.getY(pid);
                //Log.i(TAG, "x: " + x + "y: " + y);
                nativeTouch(x, y);
                //moving = false;
            }
        }

        return true;
    }

    EnjRenderer mRenderer;

    //private static native void nativePause();
    private static native void nativeTouch(float x, float y);
    private static native void nativeDown(float x, float y);
    private static native void nativeMove(float x, float y);
}

class EnjRenderer implements GLSurfaceView.Renderer {
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        nativeInit();
    }

    public void onSurfaceChanged(GL10 gl, int w, int h) {
        //gl.glViewport(0, 0, w, h);
        nativeResize(w, h);
    }

    public void onDrawFrame(GL10 gl) {
        nativeRender();
    }

    private static native void nativeInit();
    private static native void nativeResize(int w, int h);
    private static native void nativeRender();
    private static native void nativeDone();
}

