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