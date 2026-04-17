package com.microagri.app.ml

import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.Matrix
import java.nio.FloatBuffer

/**
 * On-device DeiT-Tiny plant disease classifier.
 * Loads microagri_vit_fp32.onnx and runs inference via ONNX Runtime.
 */
class DiseaseClassifier(private val env: OrtEnvironment) {

    private var session: OrtSession? = null

    data class Prediction(
        val className: String,
        val displayLabel: String,
        val confidence: Float,
        val severity: ModelConfig.Severity,
        val isHealthy: Boolean
    )

    data class ClassificationResult(
        val top1: Prediction,
        val top3: List<Prediction>,
        val latencyMs: Long
    )

    fun load() {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // Enable NNAPI on supported Snapdragon devices
            try { addNnapi() } catch (e: Exception) { /* fallback to CPU */ }
        }
        session = env.createSession(ModelConfig.CLASSIFIER_PATH.absolutePath, opts)
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        val session = session ?: throw IllegalStateException("Classifier not loaded. Call load() first.")

        val start = System.currentTimeMillis()

        // Preprocess: resize → normalize → CHW float array
        val inputTensor = preprocessBitmap(bitmap)

        // Run inference
        val inputName = session.inputNames.iterator().next()
        val inputs    = mapOf(inputName to inputTensor)
        val results   = session.run(inputs)

        val outputTensor = results.iterator().next().value as OnnxTensor
        val raw = outputTensor.value
        val logits: FloatArray = when {
            raw is Array<*> -> @Suppress("UNCHECKED_CAST") (raw as Array<FloatArray>)[0]
            raw is FloatArray -> raw
            else -> throw IllegalStateException("Unexpected classifier output: ${raw?.javaClass?.name}")
        }
        inputTensor.close()
        results.close()

        val latency = System.currentTimeMillis() - start

        // Softmax
        val maxLogit = logits.max()
        val expLogits = logits.map { Math.exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expLogits.sum()
        val probs = expLogits.map { it / sumExp }.toFloatArray()

        // Top-3
        val top3Indices = probs.indices.sortedByDescending { probs[it] }.take(3)
        val top3 = top3Indices.map { idx ->
            val cls = ModelConfig.CLASS_NAMES[idx]
            Prediction(
                className    = cls,
                displayLabel = ModelConfig.DISPLAY_LABELS[cls] ?: cls,
                confidence   = probs[idx],
                severity     = ModelConfig.SEVERITY_MAP[cls] ?: ModelConfig.Severity.NONE,
                isHealthy    = cls in ModelConfig.HEALTHY_CLASSES
            )
        }

        return ClassificationResult(top1 = top3[0], top3 = top3, latencyMs = latency)
    }

    private fun preprocessBitmap(bitmap: Bitmap): OnnxTensor {
        // Resize to 224x224
        val resized = Bitmap.createScaledBitmap(bitmap, ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE, true)

        val floatBuffer = FloatBuffer.allocate(1 * 3 * ModelConfig.IMG_SIZE * ModelConfig.IMG_SIZE)
        val pixels = IntArray(ModelConfig.IMG_SIZE * ModelConfig.IMG_SIZE)
        resized.getPixels(pixels, 0, ModelConfig.IMG_SIZE, 0, 0, ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE)

        // RGB channels separately (CHW format), normalize with ImageNet stats
        val mean = ModelConfig.IMAGENET_MEAN
        val std  = ModelConfig.IMAGENET_STD
        val size = ModelConfig.IMG_SIZE * ModelConfig.IMG_SIZE

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8)  and 0xFF) / 255.0f
            val b = (pixel          and 0xFF) / 255.0f
            floatBuffer.put(i,        (r - mean[0]) / std[0])  // R channel
            floatBuffer.put(i + size, (g - mean[1]) / std[1])  // G channel
            floatBuffer.put(i + size * 2, (b - mean[2]) / std[2])  // B channel
        }
        floatBuffer.rewind()

        val shape = longArrayOf(1, 3, ModelConfig.IMG_SIZE.toLong(), ModelConfig.IMG_SIZE.toLong())
        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), floatBuffer, shape)
    }

    fun close() {
        session?.close()
        session = null
    }
}
