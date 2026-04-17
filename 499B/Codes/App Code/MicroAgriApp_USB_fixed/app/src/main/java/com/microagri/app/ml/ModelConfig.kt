package com.microagri.app.ml

import android.os.Environment
import java.io.File

/**
 * Central configuration — matches pipeline_config.json from NB5.
 * Models live in /sdcard/MicroAgri/ (copied via USB from Drive).
 */
object ModelConfig {

    // Base directory on phone storage
    val BASE_DIR: File get() = File(
        Environment.getExternalStorageDirectory(), "MicroAgri"
    )

    // Model paths
    val CLASSIFIER_PATH  get() = File(BASE_DIR, "microagri_vit_fp32.onnx")
    val EMBEDDER_PATH    get() = File(BASE_DIR, "android_models/minilm-l6/model_int8.onnx")
    val GENERATOR_PATH   get() = File(BASE_DIR, "android_models/qwen2.5-0.5b/onnx/model_q4f16.onnx")
    val TOKENIZER_DIR    get() = File(BASE_DIR, "android_models/qwen2.5-0.5b")
    val WHISPER_DIR      get() = File(BASE_DIR, "android_models/whisper-tiny")
    val RAG_DB_PATH      get() = File(BASE_DIR, "rag/disease_kb.db")
    val PIPELINE_CONFIG  get() = File(BASE_DIR, "pipeline_config.json")

    // Inference config
    const val IMG_SIZE        = 224
    const val CONFIDENCE_THRESHOLD = 0.60f
    const val TOP_K_RETRIEVAL = 3
    const val MAX_NEW_TOKENS  = 250
    val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val IMAGENET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    // Class names — must match training order exactly
    val CLASS_NAMES = listOf(
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy"
    )

    val DISPLAY_LABELS = mapOf(
        "Pepper__bell___Bacterial_spot"               to "Bell Pepper — Bacterial Spot",
        "Pepper__bell___healthy"                       to "Bell Pepper — Healthy",
        "Potato___Early_blight"                        to "Potato — Early Blight",
        "Potato___Late_blight"                         to "Potato — Late Blight",
        "Potato___healthy"                             to "Potato — Healthy",
        "Tomato_Bacterial_spot"                        to "Tomato — Bacterial Spot",
        "Tomato_Early_blight"                          to "Tomato — Early Blight",
        "Tomato_Late_blight"                           to "Tomato — Late Blight",
        "Tomato_Leaf_Mold"                             to "Tomato — Leaf Mold",
        "Tomato_Septoria_leaf_spot"                    to "Tomato — Septoria Leaf Spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite"  to "Tomato — Spider Mites",
        "Tomato__Target_Spot"                          to "Tomato — Target Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus"        to "Tomato — Yellow Leaf Curl Virus",
        "Tomato__Tomato_mosaic_virus"                  to "Tomato — Mosaic Virus",
        "Tomato_healthy"                               to "Tomato — Healthy"
    )

    val SEVERITY_MAP = mapOf(
        "Pepper__bell___Bacterial_spot"               to Severity.MODERATE,
        "Pepper__bell___healthy"                       to Severity.NONE,
        "Potato___Early_blight"                        to Severity.MODERATE,
        "Potato___Late_blight"                         to Severity.SEVERE,
        "Potato___healthy"                             to Severity.NONE,
        "Tomato_Bacterial_spot"                        to Severity.MODERATE,
        "Tomato_Early_blight"                          to Severity.MODERATE,
        "Tomato_Late_blight"                           to Severity.SEVERE,
        "Tomato_Leaf_Mold"                             to Severity.MODERATE,
        "Tomato_Septoria_leaf_spot"                    to Severity.MODERATE,
        "Tomato_Spider_mites_Two_spotted_spider_mite"  to Severity.MODERATE,
        "Tomato__Target_Spot"                          to Severity.MODERATE,
        "Tomato__Tomato_YellowLeaf__Curl_Virus"        to Severity.SEVERE,
        "Tomato__Tomato_mosaic_virus"                  to Severity.MODERATE,
        "Tomato_healthy"                               to Severity.NONE
    )

    val HEALTHY_CLASSES = setOf(
        "Pepper__bell___healthy", "Potato___healthy", "Tomato_healthy"
    )

    enum class Severity { NONE, MODERATE, SEVERE }

    fun verifyModels(): List<String> {
        val missing = mutableListOf<String>()
        mapOf(
            "Classifier"  to CLASSIFIER_PATH,
            "Embedder"    to EMBEDDER_PATH,
            "RAG database" to RAG_DB_PATH,
        ).forEach { (name, file) ->
            if (!file.exists()) missing.add("$name: ${file.absolutePath}")
        }
        return missing
    }
}
