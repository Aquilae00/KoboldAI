from tensorizer import TensorSerializer
from transformers import AutoConfig, AutoModelForCausalLM
import time

def serialise_model(model, save_path):
    """Serialise the model and save the weights to the save_path"""
    try:
        serializer = TensorSerializer(save_path)
        start = time.time()
        serializer.write_module(model)
        end = time.time()
        print((f"Serialising model took {end - start} seconds"))
        serializer.close()
        return True
    except Exception as e:
        print("Serialisation failed with error: ", e)
        return False


from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor


def deserialise_saved_model(model_path, model_id, plaid=True):
    """Deserialise the model from the model_path and load into GPU memory"""

    # create a config object that we can use to init an empty model
    config = AutoConfig.from_pretrained(model_id)
    with no_init_or_tensor():
        # Load your model here using whatever class you need to initialise an empty model from a config.
        model = AutoModelForCausalLM.from_config(config)

    # Create the deserialiser object
    #   Note: plaid_mode is a flag that does a much faster deserialisation but isn't safe for training.
    #    -> only use it for inference.
    deserializer = TensorDeserializer(model_path, plaid_mode=True)

    # Deserialise the model straight into GPU (zero-copy)
    print(("Loading model"))
    deserializer.load_into_module(model)
    deserializer.close()

    return model
