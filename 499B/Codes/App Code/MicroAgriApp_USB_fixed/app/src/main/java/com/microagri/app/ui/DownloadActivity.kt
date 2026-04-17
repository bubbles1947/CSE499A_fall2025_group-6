package com.microagri.app.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.microagri.app.databinding.ActivityDownloadBinding
import com.microagri.app.utils.ModelDownloader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class DownloadActivity : AppCompatActivity() {

    private lateinit var binding: ActivityDownloadBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDownloadBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Check if Drive IDs have been filled in
        if (ModelDownloader.hasPlaceholderIds()) {
            binding.tvTitle.text      = "Setup Required"
            binding.tvSubtitle.text   = "Developer: fill in Google Drive file IDs in ModelDownloader.kt"
            binding.progressBar.visibility = View.GONE
            binding.btnRetry.visibility    = View.VISIBLE
            binding.btnRetry.text          = "Open Setup Guide"
            binding.btnRetry.setOnClickListener {
                startActivity(Intent(this, SetupActivity::class.java))
            }
            return
        }

        startDownload()
    }

    private fun startDownload() {
        binding.tvTitle.text    = "Downloading AI Models"
        binding.tvSubtitle.text = "First-time setup — this only happens once.\nRequires ~550 MB of storage."
        binding.progressBar.progress  = 0
        binding.btnRetry.visibility   = View.GONE
        binding.progressBar.visibility = View.VISIBLE

        lifecycleScope.launch {
            val result = ModelDownloader.downloadAll(this@DownloadActivity) { progress ->
                runOnUiThread {
                    binding.progressBar.progress = progress.overallPercent
                    binding.tvCurrentTask.text   = "Downloading: ${progress.taskName}" +
                        " (${progress.taskIndex + 1}/${progress.totalTasks})"

                    val mb = progress.bytesDownloaded / 1_000_000f
                    val total = if (progress.totalBytes > 0)
                        " / ${progress.totalBytes / 1_000_000f}%.1f MB".format(progress.totalBytes / 1_000_000f)
                    else ""
                    binding.tvBytes.text = "%.1f MB%s".format(mb, total)
                }
            }

            withContext(Dispatchers.Main) {
                if (result.isSuccess) {
                    binding.tvTitle.text    = "Download Complete"
                    binding.tvSubtitle.text = "All models ready. Starting app..."
                    binding.progressBar.progress = 100

                    // Restart from splash to load models
                    val intent = packageManager.getLaunchIntentForPackage(packageName)
                    intent?.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
                    startActivity(intent)
                    finish()
                } else {
                    binding.tvTitle.text    = "Download Failed"
                    binding.tvSubtitle.text = result.exceptionOrNull()?.message
                        ?: "Check your internet connection and try again."
                    binding.btnRetry.visibility = View.VISIBLE
                    binding.btnRetry.text       = "Retry"
                    binding.btnRetry.setOnClickListener { startDownload() }
                }
            }
        }
    }
}
