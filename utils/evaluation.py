# ==========================================
# Reverse Sequence to Text
# ==========================================

def sequence_to_text(sequence, tokenizer):

    reverse_index = {
        value: key
        for key, value in tokenizer.word_index.items()
    }

    words = []

    for token in sequence:

        token = int(token)

        if token == 0:
            continue

        word = reverse_index.get(token)

        if word is None:
            continue

        if word in ("<start>", "<end>"):
            continue

        words.append(word)

    return " ".join(words)


# ==========================================
# Generate BLEU From Test Dataset
# ==========================================

def generate_bleu_score(
        X_test,
        Y_test,
        input_tokenizer,
        target_tokenizer,
        translate_function,
        model_dir
):

    from nltk.translate.bleu_score import corpus_bleu

    references = []

    predictions = []

    for i in range(len(X_test)):

        # Convert integer sequence to Kharada sentence
        kharada_sentence = sequence_to_text(
            X_test[i],
            input_tokenizer
        )

        # Convert integer sequence to English sentence
        actual_translation = sequence_to_text(
            Y_test[i],
            target_tokenizer
        )

        # Model prediction
        predicted_translation = translate_function(
            kharada_sentence
        )

        references.append(
            [actual_translation.split()]
        )

        predictions.append(
            predicted_translation.split()
        )

    bleu = corpus_bleu(
        references,
        predictions
    )

    with open(
        os.path.join(model_dir, "bleu_score.txt"),
        "w"
    ) as f:

        f.write(f"{bleu:.4f}")

    return bleu