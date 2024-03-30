try:
    from lingua import Language, LanguageDetectorBuilder

    has_lingua = True

    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.ITALIAN,
                 Language.PORTUGUESE, Language.DUTCH, Language.DANISH, Language.SWEDISH, Language.NYNORSK,
                 Language.FINNISH, Language.POLISH, Language.UKRAINIAN]
    lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()
except ImportError:
    has_lingua = False


def indentify_language(text: str) -> str | None:
    lang = lang_detector.detect_language_of(text)
    return lang.iso_code_639_1.name.lower() if lang else None


def indentify_language_iso_639_3(text: str) -> str | None:
    lang = lang_detector.detect_language_of(text)
    return lang.iso_code_639_3.name.lower() if lang else None
