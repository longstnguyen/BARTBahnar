import argparse

from translation_pipeline import Translator
from config import SOLR_URL, TRANSLATOR_MODEL

def main():
    parser = argparse.ArgumentParser(description="Translate sentences using custom NLP models.")

    # Model and endpoint arguments
    parser.add_argument("--translator_model", type=str, default=TRANSLATOR_MODEL,
                        help="Path to BARTBahnar translation model (default: local checkpoint)")
    parser.add_argument("--classification_model", type=str, default="undertheseanlp/vietnamese-ner-v1.4.0a2", 
                        help="Word classification model (default: Underthesea NER v1.4.0a2)")
    parser.add_argument("--best_candidate_model", type=str, default="NlpHUST/gpt2-vietnamese", 
                        help="Best candidate selection model (default: GPT-2 Vietnamese by NlpHUST)")
    parser.add_argument("--solr_url", type=str, default=SOLR_URL,
                        help=f"Solr base URL (default: {SOLR_URL})")

    args = parser.parse_args()

    # Initialize Translator only once
    translator = Translator(args.classification_model, args.translator_model, args.best_candidate_model, args.solr_url)

    print("\nReady to translate! Enter a sentence or type 'exit' to quit.")

    # Run a loop to allow multiple translations
    while True:
        src_sentence = input("\n Enter a sentence to translate: ").strip()
        if src_sentence.lower() == "exit":
            print("Exiting.")
            break

        if not src_sentence:
            print("Input sentence cannot be empty. Please try again.")
            continue

        # Perform translation
        result = translator.translate(src_sentence)

        # Display the result
        print("\nFinal Translated Sentence:", result)

if __name__ == "__main__":
    main()
