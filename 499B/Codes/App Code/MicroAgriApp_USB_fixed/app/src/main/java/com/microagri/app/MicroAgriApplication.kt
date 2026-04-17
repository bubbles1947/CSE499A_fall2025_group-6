package com.microagri.app

import ai.onnxruntime.OrtEnvironment
import android.app.Application
import android.util.Log
import com.microagri.app.ml.DiseaseClassifier
import com.microagri.app.ml.DiagnosisRepository
import com.microagri.app.ml.ModelConfig
import com.microagri.app.rag.RagEngine
import com.microagri.app.utils.BengaliTtsManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

private const val TAG = "MicroAgriApp"

class MicroAgriApplication : Application() {

    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val ortEnv: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    lateinit var classifier: DiseaseClassifier
        private set

    lateinit var ragEngine: RagEngine
        private set

    lateinit var repository: DiagnosisRepository
        private set

    lateinit var tts: BengaliTtsManager
        private set

    var modelsLoaded = false
        private set

    var loadError: String? = null
        private set

    var onModelsReady: (() -> Unit)? = null

    var isLoadingModels = false
        private set

    override fun onCreate() {
        super.onCreate()
        tts = BengaliTtsManager(this)
        tts.init()
        // Model loading is deferred until MANAGE_EXTERNAL_STORAGE is confirmed
        // in SplashActivity, then triggered via loadModelsIfNeeded().
    }

    fun loadModelsIfNeeded() {
        if (modelsLoaded || isLoadingModels) return
        loadError = null  // reset so a retry after copying files works
        loadModels()
    }

    private fun loadModels() {
        isLoadingModels = true
        appScope.launch {
            try {
                Log.i(TAG, "Loading models from ${ModelConfig.BASE_DIR}")

                // Check all required files exist
                val missing = ModelConfig.verifyModels()
                if (missing.isNotEmpty()) {
                    loadError = "Missing model files:\n${missing.joinToString("\n")}"
                    Log.e(TAG, loadError!!)
                    return@launch
                }

                // Load classifier
                Log.i(TAG, "Loading DeiT-Tiny classifier...")
                classifier = DiseaseClassifier(ortEnv)
                classifier.load()
                Log.i(TAG, "Classifier loaded")

                // Load RAG engine
                Log.i(TAG, "Loading RAG engine...")
                ragEngine = RagEngine(this@MicroAgriApplication, ortEnv)
                ragEngine.load()
                Log.i(TAG, "RAG engine loaded")

                // Wire up repository
                repository = DiagnosisRepository(
                    context    = this@MicroAgriApplication,
                    classifier = classifier,
                    ragEngine  = ragEngine
                )

                modelsLoaded = true
                isLoadingModels = false
                Log.i(TAG, "All models ready")
                onModelsReady?.invoke()

            } catch (e: Exception) {
                isLoadingModels = false
                loadError = "Model loading failed: ${e.message}"
                Log.e(TAG, loadError!!, e)
            }
        }
    }

    override fun onTerminate() {
        super.onTerminate()
        if (::classifier.isInitialized) classifier.close()
        if (::ragEngine.isInitialized) ragEngine.close()
        tts.shutdown()
        ortEnv.close()
    }
}
