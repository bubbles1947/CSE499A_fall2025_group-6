package com.microagri.app.utils

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

private const val TAG = "ModelDownloader"
private const val PREFS_NAME = "microagri_prefs"
private const val KEY_MODELS_DOWNLOADED = "models_downloaded_v1"

/**
 * Downloads model files from Google Drive direct-download links on first launch.
 *
 * HOW TO GET DIRECT DOWNLOAD LINKS FROM GOOGLE DRIVE:
 * 1. Right-click file in Drive → Share → "Anyone with the link"
 * 2. Copy the share link: https://drive.google.com/file/d/FILE_ID/view
 * 3. Convert to direct link: https://drive.google.com/uc?export=download&id=FILE_ID
 * 4. For files >100MB add confirm token (handled automatically below)
 *
 * FILL IN YOUR FILE IDs below after uploading to Drive.
 */
object ModelDownloader {

    // ── FILL THESE IN after uploading to Google Drive ─────────────────────────
    // Get FILE_ID from the share URL: drive.google.com/file/d/FILE_ID/view
    private val DOWNLOAD_MANIFEST = listOf(
        DownloadTask(
            name       = "DeiT-Tiny Classifier",
            fileId     = "YOUR_DEIT_ONNX_FILE_ID",          // microagri_vit_fp32.onnx
            destPath   = "microagri_vit_fp32.onnx",
            expectedMb = 22
        ),
        DownloadTask(
            name       = "MiniLM Embedder (INT8)",
            fileId     = "YOUR_MINILM_INT8_FILE_ID",         // android_models/minilm-l6/model_int8.onnx
            destPath   = "android_models/minilm-l6/model_int8.onnx",
            expectedMb = 23
        ),
        DownloadTask(
            name       = "MiniLM Tokenizer vocab",
            fileId     = "YOUR_MINILM_VOCAB_FILE_ID",        // android_models/minilm-l6/vocab.txt
            destPath   = "android_models/minilm-l6/vocab.txt",
            expectedMb = 1
        ),
        DownloadTask(
            name       = "MiniLM Tokenizer config",
            fileId     = "YOUR_MINILM_TOKENIZER_CONFIG_ID",  // android_models/minilm-l6/tokenizer_config.json
            destPath   = "android_models/minilm-l6/tokenizer_config.json",
            expectedMb = 1
        ),
        DownloadTask(
            name       = "Qwen2.5-0.5B Generator",
            fileId     = "YOUR_QWEN_ONNX_FILE_ID",           // android_models/qwen2.5-0.5b/onnx/model_q4f16.onnx
            destPath   = "android_models/qwen2.5-0.5b/onnx/model_q4f16.onnx",
            expectedMb = 483
        ),
        DownloadTask(
            name       = "Qwen Tokenizer",
            fileId     = "YOUR_QWEN_TOKENIZER_FILE_ID",      // android_models/qwen2.5-0.5b/tokenizer.json
            destPath   = "android_models/qwen2.5-0.5b/tokenizer.json",
            expectedMb = 7
        ),
        DownloadTask(
            name       = "Qwen Tokenizer Config",
            fileId     = "YOUR_QWEN_TOKENIZER_CONFIG_ID",    // android_models/qwen2.5-0.5b/tokenizer_config.json
            destPath   = "android_models/qwen2.5-0.5b/tokenizer_config.json",
            expectedMb = 1
        ),
        DownloadTask(
            name       = "RAG Knowledge Base",
            fileId     = "YOUR_RAG_DB_FILE_ID",              // rag/disease_kb.db
            destPath   = "rag/disease_kb.db",
            expectedMb = 1
        ),
        DownloadTask(
            name       = "Pipeline Config",
            fileId     = "YOUR_PIPELINE_CONFIG_FILE_ID",     // pipeline_config.json
            destPath   = "pipeline_config.json",
            expectedMb = 1
        ),
    )
    // ─────────────────────────────────────────────────────────────────────────

    data class DownloadTask(
        val name: String,
        val fileId: String,
        val destPath: String,
        val expectedMb: Int
    )

    data class DownloadProgress(
        val taskName: String,
        val taskIndex: Int,
        val totalTasks: Int,
        val bytesDownloaded: Long,
        val totalBytes: Long,       // -1 if unknown
        val overallPercent: Int
    )

    fun areModelsDownloaded(context: Context): Boolean {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        if (!prefs.getBoolean(KEY_MODELS_DOWNLOADED, false)) return false
        // Double-check critical files actually exist
        return com.microagri.app.ml.ModelConfig.verifyModels().isEmpty()
    }

    fun hasPlaceholderIds(): Boolean =
        DOWNLOAD_MANIFEST.any { it.fileId.startsWith("YOUR_") }

    suspend fun downloadAll(
        context: Context,
        onProgress: (DownloadProgress) -> Unit
    ): Result<Unit> = withContext(Dispatchers.IO) {
        val baseDir = com.microagri.app.ml.ModelConfig.BASE_DIR

        for ((index, task) in DOWNLOAD_MANIFEST.withIndex()) {
            val destFile = File(baseDir, task.destPath)

            // Skip if already downloaded
            if (destFile.exists() && destFile.length() > 1024) {
                Log.i(TAG, "Skipping ${task.name} — already exists")
                onProgress(DownloadProgress(
                    taskName = task.name,
                    taskIndex = index,
                    totalTasks = DOWNLOAD_MANIFEST.size,
                    bytesDownloaded = destFile.length(),
                    totalBytes = destFile.length(),
                    overallPercent = ((index + 1) * 100 / DOWNLOAD_MANIFEST.size)
                ))
                continue
            }

            destFile.parentFile?.mkdirs()

            Log.i(TAG, "Downloading ${task.name}...")
            val result = downloadFile(
                fileId   = task.fileId,
                destFile = destFile,
                onProgress = { downloaded, total ->
                    val taskFraction   = if (total > 0) downloaded.toFloat() / total else 0f
                    val overallPercent = ((index + taskFraction) / DOWNLOAD_MANIFEST.size * 100).toInt()
                    onProgress(DownloadProgress(
                        taskName         = task.name,
                        taskIndex        = index,
                        totalTasks       = DOWNLOAD_MANIFEST.size,
                        bytesDownloaded  = downloaded,
                        totalBytes       = total,
                        overallPercent   = overallPercent
                    ))
                }
            )

            if (result.isFailure) {
                destFile.delete()
                return@withContext Result.failure(
                    Exception("Failed to download ${task.name}: ${result.exceptionOrNull()?.message}")
                )
            }
        }

        // Mark as downloaded
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putBoolean(KEY_MODELS_DOWNLOADED, true).apply()

        Result.success(Unit)
    }

    private fun downloadFile(
        fileId: String,
        destFile: File,
        onProgress: (Long, Long) -> Unit
    ): Result<Unit> {
        return try {
            // Google Drive direct download URL
            var urlStr = "https://drive.google.com/uc?export=download&id=$fileId"
            var connection = openConnection(urlStr)

            // Handle large-file confirmation page
            val contentType = connection.contentType ?: ""
            if (contentType.contains("text/html")) {
                // Extract confirm token from response cookies
                val cookies = connection.headerFields["Set-Cookie"]
                    ?.joinToString(";") ?: ""
                val token = Regex("confirm=([^;& ]+)").find(cookies)?.groupValues?.get(1)
                    ?: Regex("confirm=([^;& ]+)").find(
                        connection.inputStream.bufferedReader().readText()
                    )?.groupValues?.get(1)

                connection.disconnect()
                urlStr = "https://drive.google.com/uc?export=download&confirm=${token ?: "t"}&id=$fileId"
                connection = openConnection(urlStr)
            }

            val totalBytes = connection.contentLengthLong
            var downloadedBytes = 0L

            connection.inputStream.use { input ->
                FileOutputStream(destFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        downloadedBytes += bytesRead
                        onProgress(downloadedBytes, totalBytes)
                    }
                }
            }
            connection.disconnect()
            Log.i(TAG, "Downloaded ${destFile.name}: ${downloadedBytes / 1_000_000}MB")
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Download failed for $fileId", e)
            Result.failure(e)
        }
    }

    private fun openConnection(urlStr: String): HttpURLConnection {
        val url        = URL(urlStr)
        val connection = url.openConnection() as HttpURLConnection
        connection.connectTimeout = 30_000
        connection.readTimeout    = 60_000
        connection.setRequestProperty("User-Agent", "Mozilla/5.0")
        connection.instanceFollowRedirects = true
        connection.connect()
        return connection
    }
}
