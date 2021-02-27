import tensorflow as tf

def get_weights(model_base):
    res = []
    for layer in model_base.layers:
        res += list(layer.get_weights())
    return res

def build_tf_model():
    model_base = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    model_probs = tf.keras.Sequential([model_base, tf.keras.layers.Softmax()])

    

    return model_base, model_probs, get_weights(model_base)

def compute_loss(model_base, labels, input):
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels,
        logits=model_base(input)))


def compute_gradients(model_base, labels, input):
    with tf.GradientTape() as tape:
        loss = compute_loss(model_base, labels, input)
    grads = tape.gradient(loss, model_base.trainable_variables)

    for g in grads:
        if g is None:
            raise Exception('Gradients computation failed')
    return grads
