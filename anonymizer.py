from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain_experimental.data_anonymizer.deanonymizer_matching_strategies import (
    combined_exact_fuzzy_matching_strategy,
)
nlp_config = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_md"},
    ],
}

anonymizer = PresidioReversibleAnonymizer(
    languages_config=nlp_config, faker_seed=42)

input_text = "Sarah Johnson, born on June 15, 1985, lives at 123 Maple Street, Springfield, IL 62704. She works as a graphic designer at Creative Visions, LLC, located at 456 Oak Avenue, Springfield. Sarah’s phone number is (555) 987-6543, and her email address is sarah.johnson@email.com. She enjoys hiking, cooking, and reading science fiction novels in her free time."
print(input_text+"\n")

print("Anonymize: ")
anonymized_text = anonymizer.anonymize(input_text)
print(anonymized_text+"\n")


print("Mappings:")
print(anonymizer.deanonymizer_mapping)
print("\n")

# insert llm here

print("Deanonymize: ")
response_text = "Her name is Ryan Munoz, we can email her at jillrhodes@example.net"
print(anonymizer.deanonymize(response_text,
                             deanonymizer_matching_strategy=combined_exact_fuzzy_matching_strategy,)+"\n")
