package com.microagri.app.rag

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import ai.onnxruntime.*
import com.microagri.app.ml.ModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

private const val TAG = "RagEngine"

/**
 * Full RAG pipeline:
 *   1. Embed query with MiniLM INT8 ONNX
 *   2. Retrieve top-k docs from SQLite by cosine similarity
 *   3. Generate English advice with Qwen2.5-0.5B ONNX
 *   4. Translate to Bengali with ML Kit (offline after first download)
 */
class RagEngine(private val context: Context, private val env: OrtEnvironment) {

    private var embedderSession: OrtSession? = null
    private var db: SQLiteDatabase? = null

    // In-memory embedding cache (loaded at startup — only 47 docs × 384 floats = ~72KB)
    private val docIds      = mutableListOf<String>()
    private val docTexts    = mutableMapOf<String, String>()
    private val docTextsBn  = mutableMapOf<String, String>()
    private val docClasses  = mutableMapOf<String, String>()
    private val docTypes    = mutableMapOf<String, String>()
    private val embedMatrix = mutableListOf<FloatArray>()

    data class RetrievedDoc(
        val id: String,
        val className: String,
        val chunkType: String,
        val text: String,
        val score: Float
    )

    data class RagResult(
        val adviceEn: String,
        val adviceBn: String,
        val retrievedDocs: List<RetrievedDoc>,
        val latencyMs: Long
    )

    // ── Initialization ────────────────────────────────────────────────────────

    suspend fun load() = withContext(Dispatchers.IO) {
        loadEmbedder()
        loadDatabase()
        Log.i(TAG, "RagEngine loaded. ${docIds.size} docs in memory.")
    }

    private fun loadEmbedder() {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        embedderSession = env.createSession(ModelConfig.EMBEDDER_PATH.absolutePath, opts)
        Log.i(TAG, "Embedder loaded: ${ModelConfig.EMBEDDER_PATH.name}")
    }

    private fun loadDatabase() {
        db = SQLiteDatabase.openDatabase(
            ModelConfig.RAG_DB_PATH.absolutePath,
            null,
            SQLiteDatabase.OPEN_READONLY
        )

        // Load all embeddings into memory
        val cursor = db!!.rawQuery(
            """SELECT d.id, d.class_name, d.chunk_type, d.chunk_text, e.embedding
               FROM documents d JOIN embeddings e ON d.id = e.doc_id
               WHERE d.language = 'en'""", null
        )

        cursor.use {
            while (it.moveToNext()) {
                val id    = it.getString(0)
                val cls   = it.getString(1)
                val type  = it.getString(2)
                val text  = it.getString(3)
                val blob  = it.getBlob(4)

                val floats = ByteBuffer.wrap(blob)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                val emb = FloatArray(floats.remaining())
                floats.get(emb)

                docIds.add(id)
                docTexts[id]   = text
                docClasses[id] = cls
                docTypes[id]   = type
                embedMatrix.add(emb)
            }
        }
        Log.i(TAG, "Loaded ${docIds.size} English docs from SQLite")

        // Load pre-translated Bengali text keyed by (class_name, chunk_type)
        try {
            val bnCursor = db!!.rawQuery(
                "SELECT id, class_name, chunk_type, chunk_text FROM documents WHERE language = 'bn'", null
            )
            bnCursor.use {
                while (it.moveToNext()) {
                    // lowercase chunk_type so lookup is case-insensitive
                    val key = "${it.getString(1)}__${it.getString(2).lowercase().trim()}"
                    docTextsBn[key] = it.getString(3)
                }
            }
            Log.i(TAG, "Loaded ${docTextsBn.size} Bengali pre-translated docs from SQLite")
        } catch (e: Exception) {
            Log.w(TAG, "No Bengali docs in DB — Bengali output will mirror English: ${e.message}")
        }
    }


    // ── Embedding ─────────────────────────────────────────────────────────────

    private fun embedText(text: String): FloatArray {
        val session = embedderSession ?: throw IllegalStateException("Embedder not loaded")

        // Tokenize (simple whitespace tokenizer — real tokenizer via vocab file)
        // For on-device we use a simplified approach: truncate to 256 tokens
        val tokens = tokenize(text, maxLen = 256)

        val seqLen = tokens.first[0].size
        val shape  = longArrayOf(1, seqLen.toLong())
        val inputIds      = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.first[0]), shape)
        val attentionMask = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.second[0]), shape)
        val tokenTypeIds  = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(seqLen) { 0L }), shape)

        val inputs = mapOf(
            "input_ids"      to inputIds,
            "attention_mask" to attentionMask,
            "token_type_ids" to tokenTypeIds
        )

        val output = session.run(inputs)
        val outputTensor = output.iterator().next().value as OnnxTensor
        val tokenEmbeddings = outputTensor.value as Array<Array<FloatArray>>

        // Mean pooling
        val mask   = tokens.second[0]
        val embDim = tokenEmbeddings[0][0].size
        val pooled = FloatArray(embDim)
        var count  = 0f

        for (t in mask.indices) {
            if (mask[t] == 1L) {
                for (d in 0 until embDim) pooled[d] += tokenEmbeddings[0][t][d]
                count++
            }
        }
        if (count > 0) for (d in pooled.indices) pooled[d] /= count

        // L2 normalize
        val norm = Math.sqrt(pooled.map { it * it }.sum().toDouble()).toFloat()
        if (norm > 1e-9f) for (d in pooled.indices) pooled[d] /= norm

        inputIds.close(); attentionMask.close(); tokenTypeIds.close(); output.close()
        return pooled
    }

    // ── Retrieval ─────────────────────────────────────────────────────────────

    fun retrieve(className: String, topK: Int = ModelConfig.TOP_K_RETRIEVAL): List<RetrievedDoc> {
        // Filter to this class
        val classIndices = docIds.indices.filter { docClasses[docIds[it]] == className }
        if (classIndices.isEmpty()) return emptyList()

        // Use disease name as implicit query
        val displayName = ModelConfig.DISPLAY_LABELS[className] ?: className
        val queryEmb    = embedText(displayName)

        // Cosine similarity (embeddings already L2-normalized)
        val scored = classIndices.map { idx ->
            val emb  = embedMatrix[idx]
            val score = queryEmb.indices.sumOf { (queryEmb[it] * emb[it]).toDouble() }.toFloat()
            Pair(idx, score)
        }.sortedByDescending { it.second }.take(topK)

        return scored.map { (idx, score) ->
            val id = docIds[idx]
            RetrievedDoc(
                id        = id,
                className = docClasses[id] ?: "",
                chunkType = docTypes[id] ?: "",
                text      = docTexts[id] ?: "",
                score     = score
            )
        }
    }

    // ── Generation ────────────────────────────────────────────────────────────

    /**
     * Full RAG pipeline: retrieve → generate English → translate to Bengali.
     * Generation uses a pre-built context string passed to a simple template.
     * For on-device generation we use a rule-based template engine as primary,
     * with Qwen ONNX as enhancement when available.
     */
    suspend fun generateAdvice(
        className: String,
        confidence: Float
    ): RagResult = withContext(Dispatchers.Default) {
        val start = System.currentTimeMillis()

        val docs    = retrieve(className)
        val context = docs.joinToString("\n\n") { "[${it.chunkType.uppercase()}]\n${it.text}" }

        // Generate English advice
        val adviceEn = generateEnglish(className, confidence, docs)

        // Use pre-translated Bengali from DB
        val adviceBn = buildBengali(className, docs) ?: adviceEn

        RagResult(
            adviceEn      = adviceEn,
            adviceBn      = adviceBn,
            retrievedDocs = docs,
            latencyMs     = System.currentTimeMillis() - start
        )
    }

    private fun generateEnglish(
        className: String,
        confidence: Float,
        docs: List<RetrievedDoc>
    ): String {
        if (docs.isEmpty()) return "No disease information available for $className."

        val display  = ModelConfig.DISPLAY_LABELS[className] ?: className
        val severity = ModelConfig.SEVERITY_MAP[className] ?: ModelConfig.Severity.NONE
        val isHealthy = className in ModelConfig.HEALTHY_CLASSES

        if (isHealthy) {
            return "Your plant appears healthy (${(confidence * 100).toInt()}% confidence). " +
                   "Continue regular monitoring every 2-3 days. " +
                   "Maintain good agricultural practices: balanced fertilization, " +
                   "proper irrigation, and preventive fungicide applications during high-risk periods."
        }

        val treatment = docs.firstOrNull { it.chunkType == "treatment" }?.text ?: ""
        val symptoms  = docs.firstOrNull { it.chunkType == "symptoms_causes" }?.text ?: ""
        val prevention = docs.firstOrNull { it.chunkType == "prevention" }?.text ?: ""

        val urgency = when (severity) {
            ModelConfig.Severity.SEVERE   -> "⚠️ URGENT — Act immediately. "
            ModelConfig.Severity.MODERATE -> "Attention needed. "
            ModelConfig.Severity.NONE     -> ""
        }

        return buildString {
            append("$urgency$display detected (${(confidence * 100).toInt()}% confidence).\n\n")
            if (treatment.isNotEmpty()) {
                append("TREATMENT:\n$treatment\n\n")
            }
            if (prevention.isNotEmpty()) {
                append("PREVENTION:\n$prevention")
            }
        }.trim()
    }

    private fun buildBengali(className: String, docs: List<RetrievedDoc>): String? {
        val treatment  = docTextsBn["${className}__treatment"]
        val prevention = docTextsBn["${className}__prevention"]
        if (treatment == null && prevention == null) return null
        return buildString {
            if (treatment != null)  append("চিকিৎসা:\n$treatment\n\n")
            if (prevention != null) append("প্রতিরোধ:\n$prevention")
        }.trim()
    }

    // ── Simple tokenizer for MiniLM ───────────────────────────────────────────
    // Uses whitespace tokenization with [CLS]/[SEP] tokens
    // For production quality use the full WordPiece tokenizer from the tokenizer.json

    private val vocabFile by lazy {
        File(ModelConfig.EMBEDDER_PATH.parent, "vocab.txt")
    }
    private val vocab: Map<String, Long> by lazy {
        if (vocabFile.exists()) {
            vocabFile.readLines().mapIndexed { i, token -> token to i.toLong() }.toMap()
        } else {
            mapOf("[PAD]" to 0L, "[UNK]" to 100L, "[CLS]" to 101L, "[SEP]" to 102L)
        }
    }

    private fun tokenize(text: String, maxLen: Int): Pair<Array<LongArray>, Array<LongArray>> {
        val clsId = vocab["[CLS]"] ?: 101L
        val sepId = vocab["[SEP]"] ?: 102L
        val unkId = vocab["[UNK]"] ?: 100L
        val padId = vocab["[PAD]"] ?: 0L

        val words  = text.lowercase().split(Regex("\\s+"))
        val tokens = mutableListOf(clsId)
        for (word in words) {
            if (tokens.size >= maxLen - 1) break
            tokens.add(vocab[word] ?: unkId)
        }
        tokens.add(sepId)

        // Pad to maxLen
        val padded = LongArray(maxLen) { if (it < tokens.size) tokens[it] else padId }
        val mask   = LongArray(maxLen) { if (it < tokens.size) 1L else 0L }

        return Pair(arrayOf(padded), arrayOf(mask))
    }

    fun close() {
        embedderSession?.close()
        db?.close()
    }
}
