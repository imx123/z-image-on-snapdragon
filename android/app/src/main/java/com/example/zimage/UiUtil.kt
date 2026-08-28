package com.example.zimage

import android.app.Activity
import android.app.Dialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.widget.ImageView
import android.view.Window
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object UiUtil {
    private val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)

    /** Map a raw progress stage like "pipeline: denoise 3/8" to a short Chinese label. */
    fun mapStage(raw: String): String {
        val s = raw.removePrefix("pipeline: ").removePrefix("pipeline").trim()
        return when {
            s.startsWith("tokenize") -> "分词"
            s.startsWith("text encoder") -> "文本编码"
            s.startsWith("denoise") -> "去噪 " + s.removePrefix("denoise").trim()
            s.startsWith("VAE decode") -> "解码图像"
            s.contains("编译") -> "编译 " + s.substringAfter("编译").trim()
            s.contains("完成") -> "完成"
            s.isEmpty() -> "准备中…"
            else -> s
        }
    }

    fun fmtElapsed(ms: Long): String {
        val s = ms / 1000
        return "%d:%02d".format(s / 60, s % 60)
    }

    fun copyToClipboard(activity: Activity, text: String, confirm: String) {
        val cm = activity.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("zimage-diag", text))
        Toast.makeText(activity, confirm, Toast.LENGTH_SHORT).show()
    }

    /** Save bitmap into MediaStore Pictures/Z-Image; returns display path. */
    fun saveBitmap(activity: Activity, bmp: Bitmap): Pair<Uri, String>? = try {
        val name = "zimg_${ts.format(Date())}.png"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Z-Image")
        }
        val uri = activity.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)!!
        activity.contentResolver.openOutputStream(uri)!!.use { out ->
            bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        Pair(uri, "Pictures/Z-Image/$name")
    } catch (e: Exception) {
        null
    }

    /** Export plain text to MediaStore Documents/Z-Image; returns display path. */
    fun exportText(activity: Activity, text: String): String? = try {
        val name = "diag_${ts.format(Date())}.txt"
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, name)
            put(MediaStore.Downloads.MIME_TYPE, "text/plain")
            put(MediaStore.Downloads.RELATIVE_PATH, "Download/Z-Image")
        }
        val uri = activity.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)!!
        activity.contentResolver.openOutputStream(uri)!!.use { out ->
            out.write(text.toByteArray(Charsets.UTF_8))
        }
        "Download/Z-Image/$name"
    } catch (e: Exception) {
        null
    }

    /** Fullscreen image preview dialog. */
    fun showPreview(activity: Activity, bmp: Bitmap) {
        val dialog = Dialog(activity, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
        val view = LayoutInflater.from(activity).inflate(R.layout.dialog_preview, null)
        val img = view.findViewById<ImageView>(R.id.previewImage)
        img.setImageBitmap(bmp)
        view.findViewById<View>(R.id.previewClose).setOnClickListener { dialog.dismiss() }
        dialog.setContentView(view)
        dialog.show()
    }
}
