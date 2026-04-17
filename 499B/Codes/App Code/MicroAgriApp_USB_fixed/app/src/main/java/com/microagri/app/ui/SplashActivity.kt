package com.microagri.app.ui

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.microagri.app.MicroAgriApplication
import com.microagri.app.R
import com.microagri.app.ml.ModelConfig

class SplashActivity : AppCompatActivity() {

    private val handler = Handler(Looper.getMainLooper())
    private var permissionDialogShown = false
    private var routeStarted = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)
    }

    override fun onResume() {
        super.onResume()
        if (routeStarted) return

        if (hasStoragePermission()) {
            routeStarted = true
            handler.postDelayed({ route() }, 500)
        } else if (!permissionDialogShown) {
            permissionDialogShown = true
            showStoragePermissionDialog()
        } else {
            // Returned from Settings without granting — allow dialog to show again
            permissionDialogShown = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
    }

    private fun hasStoragePermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else {
            true
        }
    }

    private fun showStoragePermissionDialog() {
        AlertDialog.Builder(this)
            .setTitle("Storage Permission Required")
            .setMessage(
                "MicroAgri needs access to all files to read the AI model files " +
                "you copied to /sdcard/MicroAgri/.\n\n" +
                "Tap 'Allow' → enable 'Allow access to manage all files'."
            )
            .setCancelable(false)
            .setPositiveButton("Allow") { _, _ -> openManageStorageSettings() }
            .setNegativeButton("Exit") { _, _ -> finish() }
            .show()
    }

    private fun openManageStorageSettings() {
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

    private fun route() {
        val missing = ModelConfig.verifyModels()
        if (missing.isNotEmpty()) {
            startActivity(Intent(this, SetupActivity::class.java).apply {
                putStringArrayListExtra("missing", ArrayList(missing))
            })
            finish()
            return
        }

        val app = application as MicroAgriApplication
        app.loadModelsIfNeeded()
        pollForModels(app, timeoutMs = 60_000L)
    }

    private fun pollForModels(app: MicroAgriApplication, timeoutMs: Long, elapsedMs: Long = 0L) {
        when {
            app.modelsLoaded -> {
                startActivity(Intent(this, MainActivity::class.java))
                finish()
            }
            app.loadError != null -> {
                startActivity(Intent(this, SetupActivity::class.java).apply {
                    putExtra("error", app.loadError)
                })
                finish()
            }
            elapsedMs >= timeoutMs -> {
                startActivity(Intent(this, SetupActivity::class.java).apply {
                    putExtra("error", "Model loading timed out. Check that all files are in /sdcard/MicroAgri/.")
                })
                finish()
            }
            else -> {
                handler.postDelayed({ pollForModels(app, timeoutMs, elapsedMs + 500) }, 500)
            }
        }
    }
}
