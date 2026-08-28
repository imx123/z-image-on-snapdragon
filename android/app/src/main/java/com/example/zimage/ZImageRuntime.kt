package com.example.zimage

import android.graphics.Bitmap
import android.graphics.Color

class ZImageRuntime(modelRoot: String, backend: String = "htp") {
    private val handle = nativeCreate(modelRoot, backend)
    fun status(): String = nativeStatus(handle)
    fun generate(prompt: String, width: Int, height: Int, steps: Int, seed: Long = 42): String =
        nativeGenerate(handle, prompt, width, height, steps, seed)
    /** Raw RGB888 (width*height*3) of the last generated image, or null. */
    fun lastImage(width: Int, height: Int): Bitmap? {
        val rgb = nativeGetLastImage(handle) ?: return null
        if (rgb.size < width * height * 3) return null
        val px = IntArray(width * height)
        var o = 0
        for (i in px.indices) {
            val r = rgb[o].toInt() and 0xFF
            val g = rgb[o + 1].toInt() and 0xFF
            val b = rgb[o + 2].toInt() and 0xFF
            px[i] = Color.rgb(r, g, b)
            o += 3
        }
        return Bitmap.createBitmap(px, width, height, Bitmap.Config.ARGB_8888)
    }
    fun clearImage() = nativeClearLastImage(handle)
    fun vaeProbe(): String = nativeVaeProbe(handle)
    fun transformerProbe(latent: FloatArray, timestep: Float, capFeats: FloatArray, capMask: BooleanArray): String =
        nativeTransformerProbe(handle, latent, timestep, capFeats, capMask)
    fun textEncoderProbe(inputIds: IntArray, attentionMask: BooleanArray): String =
        nativeTextEncoderProbe(handle, inputIds, attentionMask)
    fun pollUtilization(): String = nativePollUtilization(handle)
    protected fun finalize() { nativeDestroy(handle) }
    private external fun nativeCreate(modelRoot: String, backend: String): Long
    private external fun nativeStatus(handle: Long): String
    private external fun nativeGenerate(handle: Long, prompt: String, width: Int, height: Int, steps: Int, seed: Long): String
    private external fun nativeGetLastImage(handle: Long): ByteArray?
    private external fun nativeClearLastImage(handle: Long)
    private external fun nativeVaeProbe(handle: Long): String
    private external fun nativeTransformerProbe(handle: Long, latent: FloatArray, timestep: Float, capFeats: FloatArray, capMask: BooleanArray): String
    private external fun nativeTextEncoderProbe(handle: Long, inputIds: IntArray, attentionMask: BooleanArray): String
    private external fun nativePollUtilization(handle: Long): String
    private external fun nativeDestroy(handle: Long)
    companion object { init { System.loadLibrary("zimage_runtime") } }
}
