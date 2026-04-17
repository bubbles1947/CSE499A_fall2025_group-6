package com.microagri.app.ml

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.graphics.Bitmap
import android.location.Location
import android.util.Log
import com.microagri.app.rag.RagEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.text.SimpleDateFormat
import java.util.*

private const val TAG = "DiagnosisRepository"

data class DiagnosisRecord(
    val id: Long = 0,
    val timestamp: String,
    val className: String,
    val displayLabel: String,
    val confidence: Float,
    val severity: String,
    val adviceEn: String,
    val adviceBn: String,
    val latitude: Double?,
    val longitude: Double?,
    val imagePath: String?
)

/**
 * Coordinates the full pipeline: classify → retrieve → generate → save history.
 */
class DiagnosisRepository(
    private val context: Context,
    private val classifier: DiseaseClassifier,
    private val ragEngine: RagEngine
) {

    private val historyDb = HistoryDatabase(context)

    data class DiagnosisResult(
        val record: DiagnosisRecord,
        val top3: List<DiseaseClassifier.Prediction>,
        val classifierLatencyMs: Long,
        val ragLatencyMs: Long
    )

    suspend fun diagnose(
        bitmap: Bitmap,
        location: Location? = null,
        language: String = "en"
    ): DiagnosisResult = withContext(Dispatchers.Default) {

        // Step 1: Classify
        Log.i(TAG, "Running classifier...")
        val classResult = classifier.classify(bitmap)
        Log.i(TAG, "Class: ${classResult.top1.className} (${classResult.top1.confidence})")

        // Step 2: RAG
        Log.i(TAG, "Running RAG pipeline...")
        val ragResult = ragEngine.generateAdvice(
            className  = classResult.top1.className,
            confidence = classResult.top1.confidence
        )
        Log.i(TAG, "RAG done in ${ragResult.latencyMs}ms")

        // Step 3: Save to history
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
            .format(Date())

        val record = DiagnosisRecord(
            timestamp    = timestamp,
            className    = classResult.top1.className,
            displayLabel = classResult.top1.displayLabel,
            confidence   = classResult.top1.confidence,
            severity     = classResult.top1.severity.name,
            adviceEn     = ragResult.adviceEn,
            adviceBn     = ragResult.adviceBn,
            latitude     = location?.latitude,
            longitude    = location?.longitude,
            imagePath    = null // set after saving image externally
        )

        val savedRecord = historyDb.insert(record)

        DiagnosisResult(
            record              = savedRecord,
            top3                = classResult.top3,
            classifierLatencyMs = classResult.latencyMs,
            ragLatencyMs        = ragResult.latencyMs
        )
    }

    fun getHistory(): List<DiagnosisRecord> = historyDb.getAll()

    fun getHistoryForClass(className: String): List<DiagnosisRecord> =
        historyDb.getByClass(className)
}

// ── Local history database ────────────────────────────────────────────────────

class HistoryDatabase(context: Context) :
    SQLiteOpenHelper(context, "diagnosis_history.db", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                class_name   TEXT NOT NULL,
                display_label TEXT NOT NULL,
                confidence   REAL NOT NULL,
                severity     TEXT NOT NULL,
                advice_en    TEXT,
                advice_bn    TEXT,
                latitude     REAL,
                longitude    REAL,
                image_path   TEXT
            )
        """)
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_class ON history(class_name)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_ts    ON history(timestamp)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS history")
        onCreate(db)
    }

    fun insert(record: DiagnosisRecord): DiagnosisRecord {
        val cv = ContentValues().apply {
            put("timestamp",     record.timestamp)
            put("class_name",    record.className)
            put("display_label", record.displayLabel)
            put("confidence",    record.confidence)
            put("severity",      record.severity)
            put("advice_en",     record.adviceEn)
            put("advice_bn",     record.adviceBn)
            record.latitude?.let  { put("latitude",  it) }
            record.longitude?.let { put("longitude", it) }
            record.imagePath?.let { put("image_path", it) }
        }
        val id = writableDatabase.insert("history", null, cv)
        return record.copy(id = id)
    }

    fun getAll(): List<DiagnosisRecord> {
        val cursor = readableDatabase.rawQuery(
            "SELECT * FROM history ORDER BY timestamp DESC", null
        )
        return cursor.use { parseCursor(it) }
    }

    fun getByClass(className: String): List<DiagnosisRecord> {
        val cursor = readableDatabase.rawQuery(
            "SELECT * FROM history WHERE class_name=? ORDER BY timestamp DESC",
            arrayOf(className)
        )
        return cursor.use { parseCursor(it) }
    }

    private fun parseCursor(cursor: android.database.Cursor): List<DiagnosisRecord> {
        val list = mutableListOf<DiagnosisRecord>()
        while (cursor.moveToNext()) {
            list.add(DiagnosisRecord(
                id           = cursor.getLong(cursor.getColumnIndexOrThrow("id")),
                timestamp    = cursor.getString(cursor.getColumnIndexOrThrow("timestamp")),
                className    = cursor.getString(cursor.getColumnIndexOrThrow("class_name")),
                displayLabel = cursor.getString(cursor.getColumnIndexOrThrow("display_label")),
                confidence   = cursor.getFloat(cursor.getColumnIndexOrThrow("confidence")),
                severity     = cursor.getString(cursor.getColumnIndexOrThrow("severity")),
                adviceEn     = cursor.getString(cursor.getColumnIndexOrThrow("advice_en")) ?: "",
                adviceBn     = cursor.getString(cursor.getColumnIndexOrThrow("advice_bn")) ?: "",
                latitude     = cursor.getDouble(cursor.getColumnIndexOrThrow("latitude")).takeIf { !cursor.isNull(cursor.getColumnIndexOrThrow("latitude")) },
                longitude    = cursor.getDouble(cursor.getColumnIndexOrThrow("longitude")).takeIf { !cursor.isNull(cursor.getColumnIndexOrThrow("longitude")) },
                imagePath    = cursor.getString(cursor.getColumnIndexOrThrow("image_path"))
            ))
        }
        return list
    }
}
