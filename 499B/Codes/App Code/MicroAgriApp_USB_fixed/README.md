# MicroAgri Android App — USB Version

This version loads models directly from phone storage.
No internet required at any point — not even for first setup.

---

## Step 1 — Copy models to your phone via USB

1. Connect phone to PC via USB cable
2. On your phone: pull down notification → tap "USB" → select **File Transfer (MTP)**
3. On your PC: open File Explorer → your phone → Internal Storage
4. Create a folder called `MicroAgri` at the root of Internal Storage
5. Copy these files from your PC (they're in your Google Drive CSE499B folder):

```
MicroAgri/
├── microagri_vit_fp32.onnx                  (22 MB)
├── pipeline_config.json
├── android_models/
│   ├── minilm-l6/
│   │   ├── model_int8.onnx                  (23 MB)
│   │   ├── vocab.txt
│   │   └── tokenizer_config.json
│   └── qwen2.5-0.5b/
│       ├── onnx/model_q4f16.onnx            (483 MB)
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── rag/
    └── disease_kb.db                        (0.2 MB)
```

Total: ~530 MB. Takes 3-5 minutes over USB.

---

## Step 2 — Open in Android Studio

File → Open → select MicroAgriApp_USB/ → wait for Gradle sync.

---

## Step 3 — Build and Run

Enable USB Debugging on your phone → connect → click Run.
App launches immediately, no internet, no download screen.

If files are missing the SetupActivity shows exactly what to copy.

---

## vs Drive-download version

| | USB version | Drive-download version |
|---|---|---|
| Internet needed | Never | Once (first launch ~530MB) |
| Setup | Copy via USB manually | Automatic |
| Bengali translation | Pre-translated in RAG DB | ML Kit on-device |
| Best for | Capstone demo / dev | Production |
