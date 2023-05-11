from typing import Any, Dict, List, Optional
from cog import BasePredictor, Input
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import aiserver2

# Shorthand identifier for a transformers model.
# See https://huggingface.co/models?library=transformers for a list of models.

class Predictor(BasePredictor):
    def setup(self):
        aiserver2.general_startup()
        aiserver2.patch_transformers()
        aiserver2.load_model(**{"initial_load": True})

    def predict(
        self,
        prompt: str = Input(description=f"Text prompt to send to the model."),
        n: int = Input(
            description="Number of output sequences to generate", default=1, ge=1, le=5
        ),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=50,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
    ) -> Dict[Any,Any]:
        schema = aiserver2.GenerationInputSchema
        schema.prompt = prompt
        schema.use_story = False
        schema.use_memory = False
        schema.use_authors_note = False
        schema.use_world_info = False
        schema.max_context_length = 1402
        schema.max_length = max_length
        schema.rep_pen = 1.1
        schema.rep_pen_range = 1024
        schema.rep_pen_slope = 0.9
        schema.temperature = temperature
        schema.tfs = 0.9
        schema.top_a = 0
        schema.top_k = 0
        schema.top_p = top_p
        schema.typical = 1
        schema.sampler_order = [6, 0, 1, 2, 3, 4, 5]

        result = aiserver2._generate_text(schema)
        return result

