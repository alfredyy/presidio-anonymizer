from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain_experimental.data_anonymizer.deanonymizer_matching_strategies import (
    combined_exact_fuzzy_matching_strategy,
)
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.CHINESE]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

from faker import Faker
from presidio_anonymizer.entities import OperatorConfig


class AnonymizerEngine:
    def __init__(self):
        # Configuration for the NLP engine
        self.nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [
                # English pre-trained model
                {"lang_code": "ENGLISH", "model_name": "en_core_web_md"},
                # Chinese pre-trained model
                {"lang_code": "CHINESE", "model_name": "zh_core_web_trf"},
            ],
        }
        # Create the anonymizer with the configuration
        self.anonymizer = PresidioReversibleAnonymizer(
            languages_config=self.nlp_config, faker_seed=42
        )

        # Add custom recognizer (e.g., for NRIC/FIN recognition)
        spacy_recognizer = SpacyRecognizer(
            supported_entities="SG_NRIC_FIN"
        )
        self.anonymizer.add_recognizer(spacy_recognizer)
    

    def anonymize(self, text):
        """Anonymize the provided text."""
        language = detector.detect_language_of(text).name
        anonymized_text = self.anonymizer.anonymize(text, language=language)
        return anonymized_text

    def deanonymize(self, text):
        """Deanonymize the provided text."""
        deanonymized_text = self.anonymizer.deanonymize(
            text,
            deanonymizer_matching_strategy=combined_exact_fuzzy_matching_strategy,
        )
        return deanonymized_text
