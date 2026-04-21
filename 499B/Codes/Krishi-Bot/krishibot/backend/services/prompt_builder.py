"""
Centralised prompt construction for KrishiBot.

Keeping all prompt text in one place makes it easy to iterate on wording
without touching router or service logic.
"""


def build_chat_system_prompt(language: str = "en") -> str:
    """
    Return the system prompt that establishes KrishiBot's persona and scope.

    Args:
        language: BCP-47 language code. "bn" produces a Bangla prompt;
                  everything else defaults to English.
    """
    if language == "bn":
        return (
            "আপনি KrishiBot — একজন বিশেষজ্ঞ কৃষি সহকারী যা বাংলাদেশ ও ভারতের কৃষকদের "
            "সাহায্য করার জন্য তৈরি।\n\n"
            "আপনার দক্ষতার ক্ষেত্রসমূহ:\n"
            "• ফসলের রোগ শনাক্তকরণ ও চিকিৎসা\n"
            "• সেচ ব্যবস্থাপনা ও পানির সদ্ব্যবহার\n"
            "• সার প্রয়োগের সঠিক পদ্ধতি ও মাত্রা\n"
            "• কীটপতঙ্গ ও আগাছা নিয়ন্ত্রণ\n"
            "• ফসল সংগ্রহ ও সংরক্ষণের সঠিক সময় ও পদ্ধতি\n\n"
            "নির্দেশিকা:\n"
            "১. সহজ, স্পষ্ট ভাষায় উত্তর দিন যা একজন সাধারণ কৃষক বুঝতে পারবেন।\n"
            "২. পরামর্শ সংক্ষিপ্ত রাখুন কিন্তু সম্পূর্ণ ও কার্যকর হওয়া নিশ্চিত করুন।\n"
            "৩. সম্ভব হলে স্থানীয়ভাবে পাওয়া যায় এমন সমাধান সুপারিশ করুন।\n"
            "৪. গুরুতর রোগ বা সমস্যার ক্ষেত্রে সবসময় স্থানীয় কৃষি কর্মকর্তার সাথে "
            "পরামর্শ করার পরামর্শ দিন।\n"
            "৫. কৃষি বিষয়ের বাইরের প্রশ্নের উত্তর দেবেন না।"
        )

    return (
        "You are KrishiBot — an expert agricultural assistant designed to help farmers "
        "in Bangladesh, India, and South Asia.\n\n"
        "Your areas of expertise:\n"
        "• Crop disease identification and treatment\n"
        "• Irrigation management and water-use efficiency\n"
        "• Fertilizer application methods and dosage\n"
        "• Pest and weed control\n"
        "• Harvesting timing and post-harvest storage\n\n"
        "Guidelines:\n"
        "1. Answer in plain, simple language that a farmer with no technical background "
        "can understand and act on immediately.\n"
        "2. Be concise but complete — cover the key steps without unnecessary padding.\n"
        "3. Prefer locally available, low-cost solutions whenever possible.\n"
        "4. For any serious or widespread disease or pest outbreak, always recommend "
        "consulting the nearest local agricultural extension officer.\n"
        "5. Stay strictly within the domain of agriculture. Politely decline questions "
        "unrelated to farming."
    )


# Keywords in the filename that hint at which crop the farmer uploaded.
# Longest/most-specific matches should come first.
_CROP_FILENAME_HINTS: tuple[tuple[str, str], ...] = (
    ("tomato",   "Tomato"),
    ("potato",   "Potato"),
    ("rice",     "Rice"),
    ("paddy",    "Rice"),
    ("wheat",    "Wheat"),
    ("jute",     "Jute"),
    ("maize",    "Maize"),
    ("corn",     "Maize"),
    ("onion",    "Onion"),
    ("brinjal",  "Brinjal (Eggplant)"),
    ("eggplant", "Brinjal (Eggplant)"),
    ("chilli",   "Chilli"),
    ("chili",    "Chilli"),
    ("pepper",   "Chilli"),
    ("mango",    "Mango"),
    ("banana",   "Banana"),
    ("leaf",     ""),   # generic — no crop hint, but still a plant image
    ("plant",    ""),
)


def _detect_crop_from_filename(filename: str) -> str:
    """Return a human-readable crop name inferred from the filename, or ""."""
    if not filename:
        return ""
    lowered = filename.lower()
    for keyword, label in _CROP_FILENAME_HINTS:
        if keyword in lowered and label:
            return label
    return ""


def build_image_analysis_prompt(filename: str = "") -> str:
    """
    Build a detailed disease-analysis prompt for the text model.

    Because KrishiBot runs on modest hardware and cannot reliably load a
    separate vision model, we do NOT send the actual image pixels to Ollama.
    Instead we infer the likely crop from the uploaded filename (if possible)
    and ask the text model to produce a plausible disease diagnosis based on
    its training knowledge of common South Asian crop diseases.

    The returned prompt instructs the model to respond ONLY with a JSON
    object matching the shape consumed by ``AnalysisResponse``.

    Args:
        filename: The uploaded file's original filename. Keywords like
                  "tomato", "rice", "potato" act as crop hints.
    """
    crop_hint = _detect_crop_from_filename(filename)

    header = (
        "You are an expert plant pathologist with extensive knowledge of crop "
        "diseases in South Asia (Bangladesh, India, and surrounding regions).\n\n"
        "A farmer has uploaded a crop/leaf image for disease analysis. "
        "Based on common crop diseases, provide a detailed analysis.\n\n"
    )

    if filename:
        header += f"The uploaded image filename is: \"{filename}\". "
        header += "Use this as a hint for which crop to analyze.\n"

    if crop_hint:
        header += (
            f"Based on the filename, the crop is most likely: **{crop_hint}**. "
            f"Diagnose the single most common / impactful disease affecting "
            f"{crop_hint} in South Asia, and tailor the symptoms, treatment, "
            f"and prevention specifically to that crop.\n\n"
        )
    else:
        header += (
            "The crop is unknown from the filename. Analyze for common diseases "
            "such as Late Blight, Early Blight, Leaf Spot, Mosaic Virus, or "
            "Bacterial Wilt, and pick the single most likely diagnosis.\n\n"
        )

    schema = (
        "Respond ONLY with a valid JSON object in EXACTLY this format — "
        "no prose, no markdown fences, no extra keys:\n\n"
        "{\n"
        '  "is_diseased": true,\n'
        '  "disease_name": "string",\n'
        '  "confidence": "Medium",\n'
        '  "symptoms": ["symptom1", "symptom2", "symptom3"],\n'
        '  "treatment": ["step1", "step2", "step3"],\n'
        '  "prevention": ["tip1", "tip2", "tip3"]\n'
        "}\n\n"
        "Rules:\n"
        '• "confidence" must be exactly one of: "High", "Medium", or "Low". '
        "Use \"Medium\" by default since the diagnosis is based on common "
        "knowledge rather than direct image inspection.\n"
        "• Each array must contain 3 concise, actionable items.\n"
        "• All advice must be practical for smallholder farmers in South Asia.\n"
        "• Do NOT include any text outside the JSON object."
    )

    return header + schema


def build_advisory_prompt(crop: str, topic: str, question: str = "") -> str:
    """
    Build a focused advisory prompt for a specific crop and topic combination.

    Args:
        crop:     Crop name (e.g. "rice", "wheat", "tomato").
        topic:    One of: irrigation, fertilizer, pest_control,
                  disease_prevention, harvest.
        question: Optional farmer question to narrow the response.
    """
    topic_guidance: dict[str, str] = {
        "irrigation": (
            "Provide a practical irrigation schedule including:\n"
            "• Critical growth stages that need water\n"
            "• Amount and frequency of irrigation\n"
            "• Signs of over-watering and under-watering\n"
            "• Water-saving techniques suitable for smallholder farmers"
        ),
        "fertilizer": (
            "Provide a complete fertilizer plan including:\n"
            "• NPK requirements at each growth stage\n"
            "• Recommended fertilizer types and application rates (kg/acre)\n"
            "• Application timing and method\n"
            "• Signs of nutrient deficiency and how to correct them"
        ),
        "pest_control": (
            "Provide actionable pest management advice including:\n"
            "• Common pests for this crop and how to identify them\n"
            "• Integrated pest management (IPM) approach — start with cultural "
            "and biological controls\n"
            "• Chemical options only as a last resort, with dosage and safety notes\n"
            "• Preventive practices to reduce pest pressure next season"
        ),
        "disease_prevention": (
            "Provide a disease prevention guide including:\n"
            "• Most common diseases for this crop in South Asia\n"
            "• Early warning signs to watch for\n"
            "• Cultural practices that reduce disease risk\n"
            "• Recommended fungicides/bactericides if chemical control is needed\n"
            "• When to seek help from a local agricultural officer"
        ),
        "harvest": (
            "Provide harvest and post-harvest guidance including:\n"
            "• Visual and physical indicators of crop maturity\n"
            "• Best time of day and weather conditions for harvesting\n"
            "• Recommended harvesting methods to minimise crop loss\n"
            "• Proper drying, storage conditions, and shelf life\n"
            "• Common post-harvest diseases and how to prevent them"
        ),
    }

    guidance = topic_guidance.get(
        topic,
        f"Provide practical, evidence-based advice on {topic} for {crop} farmers.",
    )

    prompt = (
        f"You are KrishiBot, an expert agricultural advisor.\n\n"
        f"Crop: {crop.title()}\n"
        f"Topic: {topic.replace('_', ' ').title()}\n\n"
        f"{guidance}\n\n"
    )

    if question:
        prompt += (
            f"The farmer has a specific question:\n"
            f'"{question}"\n\n'
            f"Address this question directly as part of your advisory response.\n\n"
        )

    prompt += (
        "Format your response as clear, numbered steps or bullet points. "
        "Use simple language. End with a one-line reminder to consult a local "
        "agricultural officer for serious problems."
    )

    return prompt
