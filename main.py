from pydantic import BaseModel
import aiserver3

aiserver3.general_startup()
aiserver3.patch_transformers()
aiserver3.load_model(**{'initial_load':True})

class Item(BaseModel):
    prompt: str
    n: int
    max_length: int
    temperature: float
    top_p: float

def predict(item, run_id, logger):
    item = Item(**item)

    ##Do something with parameters from item

    schema = aiserver3.GenerationInputSchema
    schema.prompt = item.prompt
    schema.use_story = False
    schema.use_memory = False
    schema.use_authors_note = False
    schema.use_world_info = False
    schema.max_context_length = 1402
    schema.max_length = item.max_length
    schema.rep_pen = 1.1
    schema.rep_pen_range = 1024
    schema.rep_pen_slope = 0.9
    schema.temperature = item.temperature
    schema.tfs = 0.9
    schema.top_a = 0
    schema.top_k = 0
    schema.top_p = item.top_p 
    schema.typical = 1
    schema.sampler_order = [6, 0, 1, 2, 3, 4, 5]
    schema.n = item.n

    result = aiserver3._generate_text(schema)
    return result
