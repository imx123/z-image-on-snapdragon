package com.example.zimage

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * App-sandbox GPU feasibility probe using the same LiteRT GPU delegate family
 * that google-ai-edge/gallery pulls in via tflite-gpu. Unlike libQnnGpu, the
 * delegate links against public libEGL/libGLESv3 and only optionally dlopens
 * libOpenCL, so it does not require /dev/kgsl-3d0 from the app namespace.
 */
object GpuProbe {
    private const val TAG = "zimage-gpuprobe"

    private fun floatTensor(shape: IntArray, value: Float): Any {
        // Build real Java primitive multi-dim arrays (float[][], float[][][], ...).
        // TFLite's reflection-based tensor conversion rejects Kotlin
        // Array<Array<FloatArray>> because its runtime type is Object[][].
        val array = java.lang.reflect.Array.newInstance(Float::class.javaPrimitiveType!!, *shape)
        fillFloatArray(array, value)
        return array
    }

    private fun fillFloatArray(array: Any, value: Float) {
        val length = java.lang.reflect.Array.getLength(array)
        if (array.javaClass.componentType == Float::class.javaPrimitiveType) {
            for (i in 0 until length) java.lang.reflect.Array.setFloat(array, i, value)
        } else {
            for (i in 0 until length) {
                val child = java.lang.reflect.Array.get(array, i)
                fillFloatArray(child, value)
            }
        }
    }

    private fun firstValue(tensor: Any): Float {
        var current = tensor
        while (current.javaClass.componentType != Float::class.javaPrimitiveType) {
            if (java.lang.reflect.Array.getLength(current) == 0) return 0f
            current = java.lang.reflect.Array.get(current, 0)
        }
        return if (java.lang.reflect.Array.getLength(current) == 0) 0f
        else java.lang.reflect.Array.getFloat(current, 0)
    }

    fun run(context: Context): String {
        val modelBytes = try {
            context.assets.open("gpu_probe.tflite").use { it.readBytes() }
        } catch (t: Throwable) {
            val msg = "LiteRT GPU probe: asset missing: ${t.message}"
            Log.e(TAG, msg)
            return msg
        }
        val modelBuffer =
            ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder()).apply {
                put(modelBytes)
                flip()
            }

        val compatibility = try {
            CompatibilityList().use { it.isDelegateSupportedOnThisDevice }
        } catch (t: Throwable) {
            val msg = "LiteRT GPU probe: compatibility check failed: ${t.message}"
            Log.e(TAG, msg)
            return msg
        }
        if (!compatibility) {
            val msg = "LiteRT GPU probe: delegate unsupported on this device"
            Log.e(TAG, msg)
            return msg
        }

        return try {
            val options = Interpreter.Options().apply { addDelegate(GpuDelegate()) }
            Interpreter(modelBuffer, options).use { interpreter ->
                val a = floatTensor(interpreter.getInputTensor(0).shape(), 0.5f)
                val b = floatTensor(interpreter.getInputTensor(1).shape(), 0.25f)
                val outShape = interpreter.getOutputTensor(0).shape()
                val out = floatTensor(outShape, 0f)
                // run(Object, Object) wraps its first argument in a single-element
                // Object[]; for a 2-input model we must use the multi-input API.
                interpreter.runForMultipleInputsOutputs(arrayOf(a, b), mapOf(0 to out))
                val durationNs = interpreter.lastNativeInferenceDurationNanoseconds
                val msg =
                    "LiteRT GPU probe: OK delegate=true outShape=${outShape.joinToString("x")} " +
                        "out00=${firstValue(out)} durationNs=$durationNs"
                Log.i(TAG, msg)
                msg
            }
        } catch (t: Throwable) {
            val msg = "LiteRT GPU probe: run failed: ${t.javaClass.simpleName}: ${t.message}"
            Log.e(TAG, msg, t)
            msg
        }
    }
}
