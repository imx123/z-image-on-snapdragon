package com.example.zimage

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Persistent generation history: each image is saved as a timestamped PNG
 *  under <filesDir>/zimage-runtime/images/, with an index JSON listing
 *  prompt/time so the gallery survives app restarts. */
object HistoryStore {
    data class Entry(
        val fileName: String,        // zimg_<ts>.png
        val prompt: String,
        val elapsedMs: Long,
        val createdMs: Long,
    ) {
        val displayName: String get() = fileName.removeSuffix(".png")
    }

    private const val MAX = 50
    private val items = ArrayDeque<Entry>()
    private var imagesDir: File? = null
    private var indexFile: File? = null

    private val tsFmt = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)

    /** Must be called once (onCreate) before use. Loads persisted index. */
    @Synchronized
    fun init(context: Context) {
        val dir = File(context.filesDir, "zimage-runtime/images")
        dir.mkdirs()
        imagesDir = dir
        indexFile = File(dir, "history.json")
        items.clear()
        try {
            if (indexFile!!.exists()) {
                val arr = JSONArray(indexFile!!.readText())
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    val e = Entry(o.getString("file"), o.getString("prompt"),
                        o.getLong("elapsed"), o.getLong("created"))
                    if (File(dir, e.fileName).exists()) items.addLast(e)
                }
            }
        } catch (_: Exception) { }
    }

    /** Save bitmap to app dir as timestamped PNG and prepend to history. */
    @Synchronized
    fun add(prompt: String, full: Bitmap, elapsedMs: Long) {
        val dir = imagesDir ?: return
        val now = System.currentTimeMillis()
        val name = "zimg_${tsFmt.format(Date(now))}.png"
        val file = File(dir, name)
        try {
            file.outputStream().use { full.compress(Bitmap.CompressFormat.PNG, 100, it) }
        } catch (_: Exception) { return }
        items.addFirst(Entry(name, prompt, elapsedMs, now))
        while (items.size > MAX) items.removeLast()
        saveIndex()
    }

    @Synchronized
    fun removeAt(index: Int) {
        if (index !in items.indices) return
        val e = items.removeAt(index)
        try { File(imagesDir ?: return, e.fileName).delete() } catch (_: Exception) { }
        saveIndex()
    }

    @Synchronized
    fun get(index: Int): Entry? = items.getOrNull(index)

    @Synchronized
    fun size(): Int = items.size

    @Synchronized
    fun all(): List<Entry> = items.toList()

    /** Load full-size bitmap for an entry (from app dir). */
    @Synchronized
    fun loadFull(e: Entry): Bitmap? {
        val f = File(imagesDir ?: return null, e.fileName)
        if (!f.exists()) return null
        return BitmapFactory.decodeFile(f.absolutePath)
    }

    /** Load downsampled thumbnail (max 256px, memory-safe). */
    @Synchronized
    fun loadThumb(e: Entry): Bitmap? {
        val f = File(imagesDir ?: return null, e.fileName)
        if (!f.exists()) return null
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(f.absolutePath, opts)
        var sample = 1
        while (opts.outWidth / (sample * 2) >= 256) sample *= 2
        val dec = BitmapFactory.Options().apply { inSampleSize = sample }
        return BitmapFactory.decodeFile(f.absolutePath, dec)
    }

    @Synchronized
    private fun saveIndex() {
        val dir = imagesDir ?: return
        val arr = JSONArray()
        for (e in items) {
            arr.put(JSONObject().apply {
                put("file", e.fileName)
                put("prompt", e.prompt)
                put("elapsed", e.elapsedMs)
                put("created", e.createdMs)
            })
        }
        try { File(dir, "history.json").writeText(arr.toString()) } catch (_: Exception) { }
    }
}
