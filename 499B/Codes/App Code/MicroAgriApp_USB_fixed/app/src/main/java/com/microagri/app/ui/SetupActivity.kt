package com.microagri.app.ui

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.microagri.app.MicroAgriApplication
import com.microagri.app.databinding.ActivitySetupBinding
import com.microagri.app.ml.ModelConfig

class SetupActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySetupBinding
    private val handler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySetupBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnRefresh.setOnClickListener { onCheckAgain() }
        showStatus()
    }

    override fun onResume() {
        super.onResume()
        showStatus()
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
    }

    private fun hasStoragePermission() =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R)
            Environment.isExternalStorageManager()
        else true

    private fun requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            try {
                startActivity(Intent(
                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:$packageName")
                ))
            } catch (e: Exception) {
                startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
            }
        }
    }

    private fun onCheckAgain() {
        if (!hasStoragePermission()) {
            requestStoragePermission()
            return
        }

        val missing = ModelConfig.verifyModels()
        if (missing.isNotEmpty()) {
            showStatus(missing)
            return
        }

        // Files exist — kick off loading and wait
        setLoading(true)
        val app = application as MicroAgriApplication
        app.loadModelsIfNeeded()
        pollForModels(app)
    }

    private fun pollForModels(app: MicroAgriApplication, elapsedMs: Long = 0L, timeoutMs: Long = 90_000L) {
        when {
            app.modelsLoaded -> {
                startActivity(Intent(this, MainActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK))
                finish()
            }
            app.loadError != null -> {
                setLoading(false)
                showError("Loading failed: ${app.loadError}")
            }
            elapsedMs >= timeoutMs -> {
                setLoading(false)
                showError("Timed out after ${timeoutMs / 1000}s. Models may be too large or corrupt.\nCheck logcat for details.")
            }
            else -> {
                handler.postDelayed({ pollForModels(app, elapsedMs + 500, timeoutMs) }, 500)
            }
        }
    }

    private fun setLoading(loading: Boolean) {
        binding.btnRefresh.isEnabled = !loading
        binding.btnRefresh.text = if (loading) "Loading models…" else "Check Again"
    }

    private fun showError(msg: String) {
        binding.tvInstructions.text = buildString {
            appendLine("ERROR\n")
            appendLine(msg)
            appendLine()
            appendLine("Base path: ${ModelConfig.BASE_DIR.absolutePath}")
        }
    }

    private fun showStatus(missing: List<String>? = null) {
        setLoading(false)
        val permGranted = hasStoragePermission()
        val checkedFiles = mapOf(
            "microagri_vit_fp32.onnx" to ModelConfig.CLASSIFIER_PATH,
            "minilm-l6/model_int8.onnx" to ModelConfig.EMBEDDER_PATH,
            "rag/disease_kb.db"         to ModelConfig.RAG_DB_PATH
        )

        binding.tvInstructions.text = buildString {
            appendLine("Storage permission: ${if (permGranted) "✓ Granted" else "✗ NOT GRANTED"}")
            appendLine("Base folder: ${ModelConfig.BASE_DIR.absolutePath}")
            appendLine("Base folder exists: ${ModelConfig.BASE_DIR.exists()}")
            appendLine()

            checkedFiles.forEach { (label, file) ->
                appendLine("${if (file.exists()) "✓" else "✗"} $label")
            }
            appendLine()

            if (!permGranted) {
                appendLine("Tap 'Check Again' to open permission settings.")
            } else {
                val actualMissing = missing ?: ModelConfig.verifyModels()
                if (actualMissing.isEmpty()) {
                    appendLine("All files found. Tap 'Check Again' to load the app.")
                } else {
                    appendLine("Copy missing files to Internal Storage/MicroAgri/ via USB, then tap Check Again.")
                    appendLine()
                    actualMissing.forEach { appendLine("  • $it") }
                }
            }
        }
    }
}
