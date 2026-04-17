package com.microagri.app.ui

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.microagri.app.MicroAgriApplication
import com.microagri.app.R
import com.microagri.app.databinding.ActivityResultBinding
import com.microagri.app.ml.ModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ResultActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_IMAGE_URI = "image_uri"
    }

    private lateinit var binding: ActivityResultBinding
    private var currentAdviceEn = ""
    private var currentAdviceBn = ""
    private var showingBengali = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val imageUriStr = intent.getStringExtra(EXTRA_IMAGE_URI)
        if (imageUriStr == null) {
            Toast.makeText(this, "No image provided", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        // Show image
        val uri = Uri.parse(imageUriStr)
        binding.imageView.setImageURI(uri)

        // Run diagnosis
        showLoading(true)
        runDiagnosis(uri)

        // TTS button
        binding.btnSpeak.setOnClickListener {
            val app = application as MicroAgriApplication
            if (showingBengali) {
                app.tts.speak(currentAdviceBn, currentAdviceEn)
            } else {
                app.tts.speakEnglish(currentAdviceEn)
            }
        }

        // Language toggle
        binding.btnToggleLang.setOnClickListener {
            showingBengali = !showingBengali
            updateAdviceText()
        }

        binding.btnBack.setOnClickListener { finish() }
    }

    private fun runDiagnosis(uri: Uri) {
        val app = application as MicroAgriApplication

        lifecycleScope.launch {
            try {
                // Load bitmap
                val bitmap = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it)
                    } ?: throw Exception("Could not load image")
                }

                // Run full pipeline
                val result = app.repository.diagnose(bitmap)

                // Update UI
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    displayResult(result)
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    binding.tvDisease.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun displayResult(result: com.microagri.app.ml.DiagnosisRepository.DiagnosisResult) {
        val record   = result.record
        val severity = ModelConfig.Severity.valueOf(record.severity)

        // Disease name + confidence
        binding.tvDisease.text    = record.displayLabel
        binding.tvConfidence.text = "${(record.confidence * 100).toInt()}% confidence"

        // Severity badge
        binding.tvSeverity.text = when (severity) {
            ModelConfig.Severity.SEVERE   -> "⚠ SEVERE"
            ModelConfig.Severity.MODERATE -> "⚡ MODERATE"
            ModelConfig.Severity.NONE     -> "✓ HEALTHY"
        }
        binding.tvSeverity.setBackgroundResource(when (severity) {
            ModelConfig.Severity.SEVERE   -> R.drawable.badge_red
            ModelConfig.Severity.MODERATE -> R.drawable.badge_amber
            ModelConfig.Severity.NONE     -> R.drawable.badge_green
        })

        // Top-3 predictions
        val top3Text = result.top3.joinToString("\n") {
            "${it.displayLabel}: ${(it.confidence * 100).toInt()}%"
        }
        binding.tvTop3.text = top3Text

        // Advice
        currentAdviceEn = record.adviceEn
        currentAdviceBn = record.adviceBn
        updateAdviceText()

        // Latency info
        binding.tvLatency.text =
            "Classifier: ${result.classifierLatencyMs}ms  |  RAG: ${result.ragLatencyMs}ms"

        // Show TTS button; only show language toggle if Bengali text is genuinely different
        binding.btnSpeak.visibility = View.VISIBLE
        binding.btnToggleLang.visibility =
            if (currentAdviceBn.isNotEmpty() && currentAdviceBn != currentAdviceEn)
                View.VISIBLE else View.GONE
    }

    private fun updateAdviceText() {
        if (showingBengali && currentAdviceBn.isNotEmpty()) {
            binding.tvAdvice.text        = currentAdviceBn
            binding.btnToggleLang.text   = "EN"
        } else {
            binding.tvAdvice.text        = currentAdviceEn
            binding.btnToggleLang.text   = "বাং"
        }
    }

    private fun showLoading(loading: Boolean) {
        binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        binding.tvDisease.text         = if (loading) "Analyzing..." else ""
        binding.tvAdvice.text          = if (loading) "Running AI pipeline..." else ""
        binding.btnSpeak.visibility    = View.GONE
        binding.btnToggleLang.visibility = View.GONE
    }

    override fun onDestroy() {
        super.onDestroy()
        (application as MicroAgriApplication).tts.stop()
    }
}
