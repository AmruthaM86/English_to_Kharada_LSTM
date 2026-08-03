from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model


def build_models(
    input_vocab_size,
    target_vocab_size,
    embedding_dim=256,
    latent_dim=512,
):

    # ==========================================
    # Encoder
    # ==========================================

    encoder_inputs = Input(shape=(None,), name="encoder_inputs")

    encoder_embedding = Embedding(
        input_vocab_size,
        embedding_dim,
        mask_zero=True,
        name="encoder_embedding"
    )(encoder_inputs)

    encoder_lstm = LSTM(
        latent_dim,
        return_state=True,
        name="encoder_lstm"
    )

    _, state_h, state_c = encoder_lstm(encoder_embedding)

    encoder_states = [state_h, state_c]

    # ==========================================
    # Decoder
    # ==========================================

    decoder_inputs = Input(shape=(None,), name="decoder_inputs")

    decoder_embedding_layer = Embedding(
        target_vocab_size,
        embedding_dim,
        mask_zero=True,
        name="decoder_embedding"
    )

    decoder_embedding = decoder_embedding_layer(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )

    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=encoder_states
    )

    decoder_dense = Dense(
        target_vocab_size,
        activation="softmax",
        name="decoder_dense"
    )

    decoder_outputs = decoder_dense(decoder_outputs)

    # ==========================================
    # Training Model
    # ==========================================

    training_model = Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs
    )

    training_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ==========================================
    # Encoder Inference Model
    # ==========================================

    encoder_model = Model(
        encoder_inputs,
        encoder_states
    )

    # ==========================================
    # Decoder Inference Model
    # ==========================================

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))

    decoder_states_inputs = [
        decoder_state_input_h,
        decoder_state_input_c
    ]

    decoder_embedding2 = decoder_embedding_layer(decoder_inputs)

    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        decoder_embedding2,
        initial_state=decoder_states_inputs
    )

    decoder_states = [state_h2, state_c2]

    decoder_outputs2 = decoder_dense(decoder_outputs2)

    decoder_model = Model(
        [
            decoder_inputs,
            decoder_state_input_h,
            decoder_state_input_c
        ],
        [
            decoder_outputs2,
            state_h2,
            state_c2
        ]
    )

    return (
        training_model,
        encoder_model,
        decoder_model
    )