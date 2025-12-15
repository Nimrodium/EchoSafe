from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import tensorflow as tf

def initialize_model(model: str) -> "tf.lite.Interpreter":
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    return interpreter
