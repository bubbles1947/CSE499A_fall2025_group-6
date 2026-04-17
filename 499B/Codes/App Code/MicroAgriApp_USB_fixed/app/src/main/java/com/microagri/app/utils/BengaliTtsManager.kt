package com.microagri.app.utils

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.*

private const val TAG = "BengaliTts"

class BengaliTtsManager(private val context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var isReady = false
    private var isBengaliAvailable = false

    enum class TtsState { IDLE, SPEAKING, ERROR }
    var state = TtsState.IDLE
        private set

    fun init(onReady: () -> Unit = {}) {
        tts = TextToSpeech(context, this)
    }

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) {
            Log.e(TAG, "TTS initialization failed")
            return
        }

        val bengali = Locale("bn", "BD")
        val result  = tts?.isLanguageAvailable(bengali) ?: TextToSpeech.LANG_NOT_SUPPORTED
        isBengaliAvailable = result != TextToSpeech.LANG_MISSING_DATA &&
                             result != TextToSpeech.LANG_NOT_SUPPORTED

        // Default to English; Bengali locale is set per-utterance in speak()
        tts?.setLanguage(Locale.ENGLISH)
        tts?.setSpeechRate(0.85f)
        tts?.setPitch(1.0f)

        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?)  { state = TtsState.SPEAKING }
            override fun onDone(utteranceId: String?)   { state = TtsState.IDLE }
            override fun onError(utteranceId: String?)  { state = TtsState.ERROR }
        })

        isReady = true
        Log.i(TAG, "TTS ready. Bengali available: $isBengaliAvailable")
    }

    /**
     * Speak Bengali text if Bengali TTS is available AND text is genuinely Bengali
     * (different from the English fallback). Otherwise speaks English.
     */
    fun speak(textBn: String, textEn: String) {
        if (!isReady) { Log.w(TAG, "TTS not ready"); return }
        val useBengali = isBengaliAvailable && textBn.isNotEmpty() && textBn != textEn
        if (useBengali) {
            tts?.setLanguage(Locale("bn", "BD"))
            tts?.speak(textBn, TextToSpeech.QUEUE_FLUSH, null, "bn_${System.currentTimeMillis()}")
        } else {
            speakEnglish(textEn)
        }
    }

    fun speakEnglish(text: String) {
        if (!isReady) return
        tts?.setLanguage(Locale.ENGLISH)
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "en_${System.currentTimeMillis()}")
    }

    fun stop() {
        tts?.stop()
        state = TtsState.IDLE
    }

    fun isSpeaking() = tts?.isSpeaking == true
    fun isBengaliSupported() = isBengaliAvailable

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isReady = false
    }
}
