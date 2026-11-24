"""Mots vides français (sans accents : le vectoriseur retire déjà les accents)."""

FRENCH_STOPWORDS = sorted(
    {
        "a", "ai", "au", "aux", "avec", "avez", "avoir", "ce", "ces", "cet", "cette", "ca", "cela",
        "comment", "combien", "d", "dans", "de", "des", "du", "elle", "elles", "en", "est", "et",
        "etre", "faire", "il", "ils", "j", "je", "l", "la", "le", "les", "leur", "leurs", "ma",
        "mes", "mon", "n", "ne", "nous", "on", "ou", "par", "pas", "peut", "peuvent", "pour",
        "puis", "qu", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "s", "sa",
        "se", "ses", "si", "son", "sont", "sur", "ta", "te", "tes", "ton", "tu", "un", "une",
        "vos", "votre", "vous", "y", "serai", "suis", "sera", "ete", "fait",
    }
)
