package com.example.zimage

import android.os.Bundle
import android.graphics.Bitmap
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.progressindicator.CircularProgressIndicator
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.textfield.TextInputEditText
import java.io.File

class MainActivity : AppCompatActivity() {
    private var runtime: ZImageRuntime? = null
    private var initializing = false
    @Volatile private var pollerRunning = false
    @Volatile private var generateStartMs = 0L

    private lateinit var toolbar: MaterialToolbar
    private lateinit var promptInput: TextInputEditText
    private lateinit var runButton: MaterialButton
    private lateinit var canvasCard: MaterialCardView
    private lateinit var canvasEmpty: View
    private lateinit var canvasWorking: View
    private lateinit var canvasDone: View
    private lateinit var elapsedText: TextView
    private lateinit var workingStage: TextView
    private lateinit var resultImage: android.widget.ImageView
    private lateinit var resultMeta: TextView
    private lateinit var saveButton: MaterialButton
    private lateinit var shareButton: MaterialButton
    private lateinit var historyList: RecyclerView
    private lateinit var historyHint: TextView
    private lateinit var statusBar: View
    private lateinit var statusStage: TextView
    private lateinit var statusProgress: LinearProgressIndicator
    private lateinit var diagText: TextView

    private var lastBitmap: Bitmap? = null
    private var lastMeta: String = ""
    private lateinit var settings: SettingsStore

    private enum class UiState { INITIALIZING, IDLE, GENERATING, DONE }
    private var uiState = UiState.INITIALIZING

    private fun setState(s: UiState) {
        uiState = s
        val generating = s == UiState.GENERATING
        val initializing = s == UiState.INITIALIZING
        // 按钮: IDLE 和 DONE(已出图,可改 prompt 再生成)都可点,前提文本非空
        runButton.isEnabled = (s == UiState.IDLE || s == UiState.DONE) && promptInput.text?.isNotBlank() == true
        runButton.text = if (generating) getString(R.string.generating) else getString(R.string.generate)
        promptInput.isEnabled = !generating
        canvasEmpty.visibility = if (s == UiState.IDLE) View.VISIBLE else View.GONE
        canvasWorking.visibility = if (generating || initializing) View.VISIBLE else View.GONE
        canvasDone.visibility = if (s == UiState.DONE) View.VISIBLE else View.GONE
        // 计时器只在生成中显示; 运行时加载显示纯进度环
        elapsedText.visibility = if (generating) View.VISIBLE else View.GONE
        if (initializing) workingStage.text = getString(R.string.initializing)
    }

    private fun startPoller() {
        if (pollerRunning) return
        pollerRunning = true
        Thread {
            val progressFile = File(File(filesDir, "zimage-runtime"), "progress.txt")
            while (pollerRunning) {
                try {
                    if (progressFile.exists()) {
                        val lines = progressFile.readLines()
                        if (lines.size >= 3) {
                            val stage = lines[0]
                            val pct = lines[1].toIntOrNull() ?: 0
                            val total = lines[2].toIntOrNull() ?: 100
                            val inPipeline = stage.startsWith("pipeline")
                            // 残留的 "pipeline 完成" 属于上次生成,初始化期忽略
                            val stale = inPipeline && uiState == UiState.INITIALIZING && stage.contains("完成")
                            runOnUiThread {
                                statusProgress.isIndeterminate = !inPipeline || stale
                                if (stale) {
                                    statusStage.text = getString(R.string.initializing)
                                } else if (inPipeline) {
                                    statusProgress.setProgressCompat(if (total > 0) (pct * 100 / total).coerceIn(0, 100) else 0, true)
                                    statusStage.text = UiUtil.mapStage(stage)
                                    if (uiState == UiState.GENERATING) workingStage.text = UiUtil.mapStage(stage)
                                } else if (uiState == UiState.INITIALIZING) {
                                    statusStage.text = stage.ifBlank { getString(R.string.initializing) }
                                }
                                if (uiState == UiState.GENERATING && generateStartMs > 0) {
                                    elapsedText.text = UiUtil.fmtElapsed(System.currentTimeMillis() - generateStartMs)
                                }
                            }
                        }
                    }
                } catch (_: Exception) {}
                Thread.sleep(400)
            }
        }.start()
    }

    private fun initializeRuntimeIfReady() {
        if (runtime != null || initializing) return
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R &&
            !android.os.Environment.isExternalStorageManager()) {
            statusStage.text = "需要存储权限"
            return
        }
        initializing = true
        setState(UiState.INITIALIZING)
        runButton.isEnabled = false
        Thread {
            val result = try {
                val backendFile = File(File(getExternalFilesDir(null), "probe"), "backend.txt")
                val backend = if (backendFile.exists()) backendFile.readText().trim().ifEmpty { "htp" } else "htp"
                val instance = ZImageRuntime(stageRuntimeFiles(filesDir, getExternalFilesDir(null)).absolutePath, backend)
                runtime = instance
                val st = instance.status()
                val vaeProbe = try { instance.vaeProbe() } catch (e: Throwable) { "probe failed: " + e.message }
                val gpuProbe = try { GpuProbe.run(this@MainActivity) } catch (e: Throwable) { "LiteRT GPU probe failed: " + e.message }
                st + "\n" + vaeProbe + "\n" + gpuProbe
            } catch (e: Throwable) {
                "Runtime initialization failed: " + e.message
            }
            runOnUiThread {
                initializing = false
                diagText.text = result
                if (runtime != null) {
                    // 不再把上次结果恢复到画布(用户要求); 历史图片在历史条中查看
                    setState(UiState.IDLE)
                } else setState(UiState.INITIALIZING)
                if (runtime == null) Snackbar.make(canvasCard, result, Snackbar.LENGTH_LONG).show()
            }
        }.start()
    }

    private fun stageRuntimeFiles(filesDir: File, externalFilesDir: File?): File {
        val privateRoot = File(filesDir, "zimage-runtime")
        val sourceRoot = File(externalFilesDir ?: return privateRoot, "zimage-runtime")
        val sourceLibs = File(sourceRoot, "lib/arm64-v8a")
        val targetLibs = File(privateRoot, "lib/arm64-v8a")
        if (sourceLibs.isDirectory) {
            targetLibs.mkdirs()
            sourceLibs.listFiles()?.filter { it.isFile && it.name.endsWith(".so") }?.forEach { source ->
                val target = File(targetLibs, source.name)
                if (!target.exists() || target.length() != source.length()) source.copyTo(target, overwrite = true)
            }
        }
        val sourceAssets = File(sourceRoot, "assets")
        val targetAssets = File(privateRoot, "assets")
        if (sourceAssets.isDirectory) {
            targetAssets.mkdirs()
            sourceAssets.listFiles()?.filter { it.isFile }?.forEach { source ->
                val target = File(targetAssets, source.name)
                if (!target.exists() || target.length() != source.length()) source.copyTo(target, overwrite = true)
            }
        }
        return privateRoot
    }

    private fun startGenerate() {
        val rt = runtime ?: return
        val prompt = promptInput.text?.toString()?.trim().orEmpty().ifEmpty { return }
        val steps = settings.steps
        val seed = settings.effectiveSeed
        setState(UiState.GENERATING)
        generateStartMs = System.currentTimeMillis()
        elapsedText.text = "00:00"
        Thread {
            val msg = try { rt.generate(prompt, 512, 512, steps, seed) } catch (e: Throwable) { "generate failed: " + e.message }
            val bmp = try { rt.lastImage(512, 512) } catch (_: Throwable) { null }
            runOnUiThread {
                val ok = bmp != null
                if (ok) {
                    lastBitmap = bmp
                    lastMeta = msg
                    resultImage.setImageBitmap(bmp)
                    resultMeta.text = "${prompt.take(24)} · ${(System.currentTimeMillis() - generateStartMs) / 1000}s"
                    HistoryStore.add(prompt, bmp!!, System.currentTimeMillis() - generateStartMs)
                    refreshHistory()
                    // 完成后保持 DONE 显示图; 按钮已由 setState 启用(可改 prompt 再生成)
                    setState(UiState.DONE)
                } else {
                    setState(UiState.IDLE)
                    Snackbar.make(canvasCard, msg, Snackbar.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private val historyAdapter = object : RecyclerView.Adapter<HistoryVH>() {
        override fun onCreateViewHolder(parent: android.view.ViewGroup, viewType: Int): HistoryVH {
            val v = layoutInflater.inflate(R.layout.item_history, parent, false)
            return HistoryVH(v)
        }
        override fun getItemCount(): Int = HistoryStore.size()
        override fun onBindViewHolder(h: HistoryVH, pos: Int) {
            val e = HistoryStore.get(pos) ?: return
            // 缩略图从磁盘加载(持久化历史)
            HistoryStore.loadThumb(e)?.let { h.thumb.setImageBitmap(it) }
            h.label.text = e.prompt
            // 点击: 加载全图回填画布
            h.itemView.setOnClickListener {
                val full = HistoryStore.loadFull(e)
                if (full != null) {
                    lastBitmap = full
                    lastMeta = "${e.prompt} · ${e.elapsedMs / 1000}s"
                    resultImage.setImageBitmap(full)
                    resultMeta.text = lastMeta
                    setState(UiState.DONE)
                } else Snackbar.make(canvasCard, "图片文件丢失", Snackbar.LENGTH_SHORT).show()
            }
            // 长按: 操作菜单(保存/分享/删除)
            h.itemView.setOnLongClickListener {
                showHistoryActions(e, pos)
                true
            }
        }
    }

    /** 历史项长按菜单: 保存(相册)/分享/删除 */
    private fun showHistoryActions(e: HistoryStore.Entry, pos: Int) {
        val full = HistoryStore.loadFull(e) ?: run {
            Snackbar.make(canvasCard, "图片文件丢失", Snackbar.LENGTH_SHORT).show()
            return
        }
        MaterialAlertDialogBuilder(this)
            .setTitle(e.displayName)
            .setMessage(e.prompt)
            .setPositiveButton("保存") { _, _ ->
                val saved = UiUtil.saveBitmap(this, full)
                if (saved != null) Snackbar.make(canvasCard, getString(R.string.saved_to, saved.second), Snackbar.LENGTH_LONG).show()
                else Snackbar.make(canvasCard, getString(R.string.save_failed, "写入失败"), Snackbar.LENGTH_SHORT).show()
            }
            .setNeutralButton("分享") { _, _ ->
                val saved = UiUtil.saveBitmap(this, full)
                if (saved != null) {
                    val send = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                        type = "image/png"
                        putExtra(android.content.Intent.EXTRA_STREAM, saved.first)
                        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    startActivity(android.content.Intent.createChooser(send, null))
                }
            }
            .setNegativeButton("删除") { _, _ ->
                HistoryStore.removeAt(pos)
                refreshHistory()
            }
            .show()
    }

    private fun refreshHistory() {
        historyAdapter.notifyDataSetChanged()
        historyHint.visibility = if (HistoryStore.size() > 0) View.VISIBLE else View.GONE
        historyList.visibility = if (HistoryStore.size() > 0) View.VISIBLE else View.GONE
    }

    private fun showDiagSheet() {
        val dialog = BottomSheetDialog(this)
        val view = layoutInflater.inflate(R.layout.sheet_diag, null)
        val content = view.findViewById<TextView>(R.id.diagContent)
        // Compose: probe text + backend + full jni.log
        val logFile = File(File(filesDir, "zimage-runtime"), "jni.log")
        val log = try { logFile.readText().takeLast(60000) } catch (_: Exception) { "(jni.log 不可读)" }
        val backend = try {
            File(File(getExternalFilesDir(null), "probe"), "backend.txt").readText().trim()
        } catch (_: Exception) { "?" }
        content.text = "backend: $backend\n—— probe 输出 ——\n${diagText.text}\n—— jni.log（末尾 60KB）——\n$log"
        view.findViewById<MaterialButton>(R.id.diagCopy).setOnClickListener {
            UiUtil.copyToClipboard(this, content.text.toString(), getString(R.string.diag_copied))
        }
        view.findViewById<MaterialButton>(R.id.diagExport).setOnClickListener {
            val path = UiUtil.exportText(this, content.text.toString())
            if (path != null) Snackbar.make(canvasCard, getString(R.string.diag_exported, path), Snackbar.LENGTH_LONG).show()
            else Snackbar.make(canvasCard, getString(R.string.diag_export_failed, "写入失败"), Snackbar.LENGTH_LONG).show()
        }
        dialog.setContentView(view)
        dialog.show()
    }

    private fun showSettingsDialog() {
        val dialog = com.google.android.material.dialog.MaterialAlertDialogBuilder(this).create()
        val view = layoutInflater.inflate(R.layout.dialog_settings, null)
        val resInput = view.findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.resInput)
        val stepsInput = view.findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.stepsInput)
        val randomSwitch = view.findViewById<com.google.android.material.materialswitch.MaterialSwitch>(R.id.randomSeedSwitch)
        val seedInput = view.findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.seedInput)

        resInput.setText(settings.resolutionLabel)
        stepsInput.setText(settings.steps.toString())
        randomSwitch.isChecked = settings.randomSeed
        seedInput.setText(settings.fixedSeed.toString())
        seedInput.isEnabled = !randomSwitch.isChecked
        randomSwitch.setOnCheckedChangeListener { _, checked -> seedInput.isEnabled = !checked }

        view.findViewById<com.google.android.material.button.MaterialButton>(R.id.settingsCancel).setOnClickListener { dialog.dismiss() }
        view.findViewById<com.google.android.material.button.MaterialButton>(R.id.settingsOk).setOnClickListener {
            val steps = stepsInput.text?.toString()?.toIntOrNull()?.coerceIn(1, 16) ?: 8
            val seed = seedInput.text?.toString()?.toLongOrNull() ?: 42L
            settings.steps = steps
            settings.randomSeed = randomSwitch.isChecked
            settings.fixedSeed = seed
            settings.resolutionLabel = resInput.text?.toString()?.ifBlank { "512×512" } ?: "512×512"
            Snackbar.make(canvasCard, "设置已保存：${steps} 步，种子=${if (randomSwitch.isChecked) "随机" else seed}", Snackbar.LENGTH_SHORT).show()
            dialog.dismiss()
        }
        dialog.setView(view)
        dialog.show()
    }

    override fun onResume() {
        super.onResume()
        startPoller()
        initializeRuntimeIfReady()
    }

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        settings = SettingsStore(this)
        HistoryStore.init(this)
        setContentView(R.layout.activity_main)
        toolbar = findViewById(R.id.toolbar)
        promptInput = findViewById(R.id.promptInput)
        runButton = findViewById(R.id.runButton)
        canvasCard = findViewById(R.id.canvasCard)
        canvasEmpty = findViewById(R.id.canvasEmpty)
        canvasWorking = findViewById(R.id.canvasWorking)
        canvasDone = findViewById(R.id.canvasDone)
        elapsedText = findViewById(R.id.elapsedText)
        workingStage = findViewById(R.id.workingStage)
        resultImage = findViewById(R.id.resultImage)
        resultMeta = findViewById(R.id.resultMeta)
        saveButton = findViewById(R.id.saveButton)
        shareButton = findViewById(R.id.shareButton)
        historyList = findViewById(R.id.historyList)
        historyHint = findViewById(R.id.historyHint)
        statusBar = findViewById(R.id.statusBar)
        statusStage = findViewById(R.id.statusStage)
        statusProgress = findViewById(R.id.statusProgress)
        diagText = findViewById(R.id.diagText)

        historyList.layoutManager = LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)
        historyList.adapter = historyAdapter
        // 启动时展示持久化历史(重启后图片仍在)
        refreshHistory()

        toolbar.inflateMenu(R.menu.menu_main)
        toolbar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_diag -> { showDiagSheet(); true }
                R.id.action_settings -> { showSettingsDialog(); true }
                else -> false
            }
        }

        promptInput.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: android.text.Editable?) {
                // IDLE 或 DONE(已出图)时文本变化都重新启用按钮
                if (uiState == UiState.IDLE || uiState == UiState.DONE)
                    runButton.isEnabled = !s.isNullOrBlank()
            }
        })

        runButton.setOnClickListener { startGenerate() }
        resultImage.setOnClickListener {
            lastBitmap?.let { UiUtil.showPreview(this, it) }
        }
        saveButton.setOnClickListener {
            lastBitmap?.let { bmp ->
                val saved = UiUtil.saveBitmap(this, bmp)
                if (saved != null) Snackbar.make(canvasCard, getString(R.string.saved_to, saved.second), Snackbar.LENGTH_LONG).show()
                else Snackbar.make(canvasCard, getString(R.string.save_failed, "写入失败"), Snackbar.LENGTH_SHORT).show()
            }
        }
        shareButton.setOnClickListener {
            lastBitmap?.let { bmp ->
                val saved = UiUtil.saveBitmap(this, bmp)
                if (saved != null) {
                    val send = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                        type = "image/png"
                        putExtra(android.content.Intent.EXTRA_STREAM, saved.first)
                        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    startActivity(android.content.Intent.createChooser(send, null))
                }
            }
        }

        // Edge-to-edge insets: AppBarLayout 整体下移避开状态栏, 底部状态条避开手势小白条
        val appbar = findViewById<com.google.android.material.appbar.AppBarLayout>(R.id.appbar)
        val root = findViewById<View>(R.id.scroll).parent as View
        root.setOnApplyWindowInsetsListener { v, insets ->
            val bars = insets.getInsets(
                android.view.WindowInsets.Type.statusBars()
                    or android.view.WindowInsets.Type.displayCutout()
                    or android.view.WindowInsets.Type.navigationBars())
            appbar.setPadding(0, bars.top, 0, 0)
            val lp = statusBar.layoutParams as androidx.coordinatorlayout.widget.CoordinatorLayout.LayoutParams
            lp.bottomMargin = bars.bottom
            statusBar.layoutParams = lp
            insets
        }
        setState(UiState.INITIALIZING)
    }

    override fun onDestroy() {
        super.onDestroy()
        pollerRunning = false
    }
}

private class HistoryVH(v: View) : RecyclerView.ViewHolder(v) {
    val thumb: android.widget.ImageView = v.findViewById(R.id.thumbImage)
    val label: TextView = v.findViewById(R.id.thumbPrompt)
}
