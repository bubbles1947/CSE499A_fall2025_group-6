package com.microagri.app.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.microagri.app.MicroAgriApplication
import com.microagri.app.R
import com.microagri.app.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) openCamera()
        else Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
    }

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            // Persist read permission so ResultActivity can access the URI
            contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            launchResult(it)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val app = application as MicroAgriApplication

        // Status indicator
        if (app.modelsLoaded) {
            binding.statusText.text = "Models ready"
            binding.statusIndicator.setBackgroundResource(R.drawable.circle_green)
        } else {
            binding.statusText.text = "Loading models..."
            binding.statusIndicator.setBackgroundResource(R.drawable.circle_amber)
            app.onModelsReady = {
                runOnUiThread {
                    binding.statusText.text = "Models ready"
                    binding.statusIndicator.setBackgroundResource(R.drawable.circle_green)
                }
            }
        }

        binding.btnCamera.setOnClickListener {
            if (app.modelsLoaded) checkCameraPermission()
            else Toast.makeText(this, "Models still loading...", Toast.LENGTH_SHORT).show()
        }

        binding.btnGallery.setOnClickListener {
            if (app.modelsLoaded) galleryLauncher.launch(arrayOf("image/*"))
            else Toast.makeText(this, "Models still loading...", Toast.LENGTH_SHORT).show()
        }

        binding.btnHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED -> openCamera()
            else -> cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun openCamera() {
        startActivity(Intent(this, CameraActivity::class.java))
    }

    private fun launchResult(imageUri: Uri) {
        startActivity(Intent(this, ResultActivity::class.java).apply {
            putExtra(ResultActivity.EXTRA_IMAGE_URI, imageUri.toString())
        })
    }
}
