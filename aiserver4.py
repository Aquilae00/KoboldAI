import os
from logger import logger, set_logger_verbosity, quiesce_logger
import utils
from modeling.patches import patch_transformers
import torch
import koboldai_settings
import warnings
from os import path, getcwd
import json
import gc
from eventlet import tpool
from collections import OrderedDict
from marshmallow import Schema, fields, validate, EXCLUDE
from typing import Any, Callable, TypeVar, Tuple, Union, Dict, Set, List, Optional, Type
from marshmallow.exceptions import ValidationError
import time
import fileops


class FakeSocket:
    def emit():
        return


koboldai_vars = koboldai_settings.koboldai_vars(FakeSocket())
utils.koboldai_vars = koboldai_vars

from transformers import PreTrainedTokenizerBase

old_pretrainedtokenizerbase_from_pretrained = (
    PreTrainedTokenizerBase.from_pretrained.__func__
)


@classmethod
def new_pretrainedtokenizerbase_from_pretrained(cls, *args, **kwargs):
    tokenizer = old_pretrainedtokenizerbase_from_pretrained(cls, *args, **kwargs)
    tokenizer._koboldai_header = []
    return tokenizer


PreTrainedTokenizerBase.from_pretrained = new_pretrainedtokenizerbase_from_pretrained


class Namespace:
    def __init__(self):
        self.cpu = False
        self.peft = None
        self.breakmodel_disklayers = None
        self.breakmodel_gpulayers = None
        self.breakmodel_layers = None
        self.model = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
        self.no_aria2 = None


class VarClass:
    def __init__(self):
        self.model = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
        self.noai = (
            False  # Runs the script without starting up the transformers pipeline
        )
        self.aibusy = False  # Stops submissions while the AI is working
        self.status_message = ""
        self.serverstarted = False  # Whether or not the Flask server has started
        self.lua_state = None  # Lua state of the Lua scripting system
        self.lua_koboldbridge = None  # `koboldbridge` from bridge.lua
        self.lua_kobold = None  # `kobold` from` bridge.lua
        self.lua_koboldcore = None  # `koboldcore` from bridge.lua
        self.lua_logname = ...  # Name of previous userscript that logged to terminal
        self.lua_running = (
            False  # Whether or not Lua is running (i.e. wasn't stopped due to an error)
        )
        self.abort = False  # Whether or not generation was aborted by clicking on the submit button during generation
        self.compiling = False  # If using a TPU Colab, this will be set to True when the TPU backend starts compiling and then set to False again
        self.checking = False  # Whether or not we are actively checking to see if TPU backend is compiling or not
        self.sp_changed = False  # This gets set to True whenever a userscript changes the soft prompt so that check_for_sp_change() can alert the browser that the soft prompt has changed
        self.spfilename = ""  # Filename of soft prompt to load, or an empty string if not using a soft prompt
        self.userscripts = []  # List of userscripts to load
        self.last_userscripts = (
            []
        )  # List of previous userscript filenames from the previous time userscripts were send via usstatitems
        self.corescript = "default.lua"  # Filename of corescript to load
        self.gpu_device = (
            0  # Which PyTorch device to use when using pure GPU generation
        )
        self.hascuda = True  # Whether torch has detected CUDA on the system
        self.usegpu = True  # Whether to launch pipeline with GPU support
        self.splist = []
        self.spselect = ""  # Temporary storage for soft prompt filename to load
        self.spmeta = (
            None  # Metadata of current soft prompt, or None if not using a soft prompt
        )
        self.spname = "Not in Use"  # Name of the soft prompt
        self.sp = None  # Current soft prompt tensor (as a NumPy array)
        self.sp_length = 0  # Length of current soft prompt in tokens, or 0 if not using a soft prompt
        self.has_genmod = False  # Whether or not at least one loaded Lua userscript has a generation modifier
        self.breakmodel = True  # For GPU users, whether to use both system RAM and VRAM to conserve VRAM while offering speedup compared to CPU-only
        self.bmsupported = False  # Whether the breakmodel option is supported (GPT-Neo/GPT-J/XGLM/OPT only, currently)
        self.nobreakmodel = False  # Something specifically requested Breakmodel to be disabled (For example a models config)
        self.smandelete = (
            False  # Whether stories can be deleted from inside the browser
        )
        self.smanrename = (
            False  # Whether stories can be renamed from inside the browser
        )
        self.allowsp = False  # Whether we are allowed to use soft prompts (by default enabled if we're using GPT-2, GPT-Neo or GPT-J)
        self.host = False
        self.flaskwebgui = False
        self.quiet = False  # If set will suppress any story text from being printed to the console (will only be seen on the client web page)
        self.use_colab_tpu = False  # Whether or not we're in a Colab TPU instance or Kaggle TPU instance and are going to use the TPU rather than the CPU
        self.aria2_port = 6799  # Specify the port on which aria2's RPC interface will be open if aria2 is installed (defaults to 6799)
        self.standalone = False
        self.api_tokenizer_id = None
        self.disable_set_aibusy = False
        self.disable_input_formatting = False
        self.disable_output_formatting = False
        self.full_determinism = False  # Whether or not full determinism is enabled
        self.seed_specified = False  # Whether or not the current RNG seed was specified by the user (in their settings file)
        self.rng_states = (
            {}
        )  # creates an empty dictionary to store the random number generator (RNG) states for a given seed, which is used to restore the RNG state later on
        self.seed = None  # The current RNG seed (as an int), or None if unknown
        self.alt_gen = (
            False  # Use the calc_ai_text method for generating text to go to the AI
        )
        self.cloudflare_link = ""
        self.story_loads = {}  # dict of when each story was last loaded
        self.standalone = False
        self.disable_set_aibusy = False
        self.disable_input_formatting = False
        self.disable_output_formatting = False
        self.api_tokenizer_id = None
        self.port = 5000
        self.on_colab = False
        self.horde_share = False
        self._horde_pid = None
        self.generating_image = False  # The current status of image generation
        self.image_pipeline = None
        self.summarizer = None
        self.summary_tokenizer = None
        self.keep_img_gen_in_memory = False
        self.cookies = (
            {}
        )  # cookies for colab since colab's URL changes, cookies are lost
        self.experimental_features = False
        self.seen_messages = []
        self.git_repository = ""
        self.git_branch = ""

        self.trust_remote_code = False
        self.wirmvwhtsp = False  # Whether to remove leading whitespace from WI entries
        self.widepth = 3  # How many historical actions to scan for WI hits
        self.formatoptns = {
            "frmttriminc": True,
            "frmtrmblln": False,
            "frmtrmspch": False,
            "frmtadsnsp": True,
            "singleline": False,
        }  # Container for state of formatting options
        self.frmttriminc = True
        self.frmtrmblln = False
        self.frmtrmspch = False
        self.frmtadsnsp = True
        self.singleline = False
        self.remove_double_space = True
        self.importnum = -1  # Selection on import popup list
        self.importjs = {}  # Temporary storage for import data
        self.loadselect = ""  # Temporary storage for story filename to load
        self.spselect = ""  # Temporary storage for soft prompt filename to load
        self.svowname = ""  # Filename that was flagged for overwrite confirm
        self.saveow = False  # Whether or not overwrite confirm has been displayed
        self.laststory = None  # Filename (without extension) of most recent story JSON file we loaded
        self.sid = ""  # session id for the socketio client (request.sid)
        self.username = "Default User"  # Displayed Username
        self.nopromptgen = False
        self.rngpersist = False
        self.nogenmod = False
        self.debug = False  # If set to true, will send debug information to the client for display
        self.output_streaming = True
        self.show_probs = False  # Whether or not to show token probabilities
        self.beep_on_complete = False
        self.img_gen_priority = 1
        self.show_budget = False
        self.ui_level = 2
        self.img_gen_api_url = "http://127.0.0.1:7860"
        self.img_gen_art_guide = "masterpiece, digital painting, <|>, dramatic lighting, highly detailed, trending"
        self.img_gen_negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
        self.img_gen_api_username = ""
        self.img_gen_api_password = ""
        self.img_gen_steps = 30
        self.img_gen_cfg_scale = 7.0
        self.cluster_requested_models = (
            []
        )  # The models which we allow to generate during cluster mode
        self.wigen_use_own_wi = False
        self.wigen_amount = 80
        self.screenshot_show_story_title = True
        self.screenshot_show_author_name = True
        self.screenshot_author_name = "Anonymous"
        self.screenshot_use_boring_colors = False
        self.oaiurl = ""  # OpenAI API URL
        self.revision = None
        self.oaiengines = "https://api.openai.com/v1/engines"
        self.url = (
            "https://api.inferkit.com/v1/models/standard/generate"  # InferKit API URL
        )
        self.colaburl = ""  # Ngrok url for Google Colab mode
        self.apikey = ""  # API key to use for InferKit API calls
        self.oaiapikey = ""  # API key to use for OpenAI API calls
        self.horde_api_key = "0000000000"
        self.horde_worker_name = "My Awesome Instance"
        self.horde_url = "https://horde.koboldai.net"
        self.colab = False
        self.cpu = False
        self.peft = None
        self.breakmodel_disklayers = None
        self.breakmodel_gpulayers = None
        self.breakmodel_layers = None
        self.model = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
        self.path = "/persistent-storage"
        self.savemodel = True
        self.badwordsids = [
            [6880],
            [50256],
            [42496],
            [4613],
            [17414],
            [22039],
            [16410],
            [27],
            [29],
            [38430],
            [37922],
            [15913],
            [24618],
            [28725],
            [58],
            [47175],
            [36937],
            [26700],
            [12878],
            [16471],
            [37981],
            [5218],
            [29795],
            [13412],
            [45160],
            [3693],
            [49778],
            [4211],
            [20598],
            [36475],
            [33409],
            [44167],
            [32406],
            [29847],
            [29342],
            [42669],
            [685],
            [25787],
            [7359],
            [3784],
            [5320],
            [33994],
            [33490],
            [34516],
            [43734],
            [17635],
            [24293],
            [9959],
            [23785],
            [21737],
            [28401],
            [18161],
            [26358],
            [32509],
            [1279],
            [38155],
            [18189],
            [26894],
            [6927],
            [14610],
            [23834],
            [11037],
            [14631],
            [26933],
            [46904],
            [22330],
            [25915],
            [47934],
            [38214],
            [1875],
            [14692],
            [41832],
            [13163],
            [25970],
            [29565],
            [44926],
            [19841],
            [37250],
            [49029],
            [9609],
            [44438],
            [16791],
            [17816],
            [30109],
            [41888],
            [47527],
            [42924],
            [23984],
            [49074],
            [33717],
            [31161],
            [49082],
            [30138],
            [31175],
            [12240],
            [14804],
            [7131],
            [26076],
            [33250],
            [3556],
            [38381],
            [36338],
            [32756],
            [46581],
            [17912],
            [49146],
        ]
        self.lowmem = False
        self.remote = True
        self.override_rename = True
        self.override_delete = True
        self.quiet = True
        self.noaimenu = True


# koboldai_vars = VarClass()
# utils.koboldai_vars = koboldai_vars


def general_startup(override_args=None):
    global args
    global enable_whitelist
    global allowed_ips
    import configparser

    # Figure out what git we're on if that's available
    config = configparser.ConfigParser()
    if os.path.exists(".git/config"):
        config.read(".git/config")
        koboldai_vars.git_repository = config['remote "origin"']["url"]
        for item in config.sections():
            if "branch" in item:
                koboldai_vars.git_branch = item.replace("branch ", "").replace('"', "")

        logger.info(
            "Running on Repo: {} Branch: {}".format(
                koboldai_vars.git_repository, koboldai_vars.git_branch
            )
        )

    args = VarClass()
    utils.args = Namespace()
    # load system and user settings
    for setting in ["user_settings", "system_settings"]:
        if os.path.exists("settings/{}.v2_settings".format(setting)):
            with open("settings/{}.v2_settings".format(setting), "r") as settings_file:
                getattr(koboldai_vars, "_{}".format(setting)).from_json(
                    settings_file.read()
                )

    temp = [x for x in vars(args)]
    for arg in temp:
        if arg == "path":
            if "model_path" in os.environ:
                setattr(args, arg, os.environ["model_path"])
        else:
            if arg in os.environ:
                if isinstance(getattr(args, arg), bool):
                    if os.environ[arg].lower() == "true":
                        setattr(args, arg, True)
                    else:
                        setattr(args, arg, False)
                else:
                    setattr(args, arg, os.environ[arg])

    if args.model:
        koboldai_vars.model = args.model
    koboldai_vars.revision = args.revision

    if args.colab:
        args.remote = True
        args.override_rename = True
        args.override_delete = True
        args.nobreakmodel = True
        args.quiet = True
        args.lowmem = True
        args.noaimenu = True

    if args.quiet:
        koboldai_vars.quiet = True

    if args.nobreakmodel:
        koboldai_vars.nobreakmodel = True

    if args.remote:
        koboldai_vars.host = True

    if args.trust_remote_code:
        logger.warning("EXECUTION OF UNSAFE REMOTE CODE IS ENABLED!!!")
        logger.warning("You are not protected from Model Viruses in this mode!")
        logger.warning("Exit the program now to abort execution!")
        logger.warning("Only use this mode with models that you trust and verified!")
        time.sleep(25)
        koboldai_vars.trust_remote_code = True
    if args.cpu:
        koboldai_vars.use_colab_tpu = False

    koboldai_vars.smandelete = koboldai_vars.host == args.override_delete
    koboldai_vars.smanrename = koboldai_vars.host == args.override_rename

    koboldai_vars.aria2_port = args.aria2_port or 6799

    # Now let's look to see if we are going to force a load of a model from a user selected folder
    if koboldai_vars.model == "selectfolder":
        print(
            "{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(
                colors.CYAN, colors.END
            )
        )
        modpath = fileops.getdirpath(
            getcwd() + "/persistent-storage", "Select Model Folder"
        )

        if modpath:
            # Save directory to koboldai_vars
            koboldai_vars.model = "NeoCustom"
            koboldai_vars.custmodpth = modpath
    elif args.model:
        logger.message(f"Welcome to KoboldAI!")
        logger.message(f"You have selected the following Model: {koboldai_vars.model}")
        if args.path:
            logger.message(
                f"You have selected the following path for your Model: {args.path}"
            )
            koboldai_vars.custmodpth = args.path
            koboldai_vars.colaburl = args.path + "/request"
            # Lets just use the same parameter to keep it simple

    koboldai_vars.model = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"


def unload_model():
    global model
    global generator
    global model_config
    global tokenizer

    # We need to wipe out the existing model and refresh the cuda cache
    model = None
    generator = None
    model_config = None
    koboldai_vars.online_model = ""
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="torch.distributed.reduce_op is deprecated"
            )
            for tensor in gc.get_objects():
                try:
                    if torch.is_tensor(tensor):
                        tensor.set_(
                            torch.tensor((), device=tensor.device, dtype=tensor.dtype)
                        )
                except:
                    pass
    gc.collect()
    try:
        with torch.no_grad():
            torch.cuda.empty_cache()
    except:
        pass

    # Reload our badwords
    koboldai_vars.badwordsids = koboldai_settings.badwordsids_default


def getmodelname():
    if koboldai_vars.online_model != "":
        return f"{koboldai_vars.model}/{koboldai_vars.online_model}"
    if koboldai_vars.model in (
        "NeoCustom",
        "GPT2Custom",
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    ):
        modelname = os.path.basename(os.path.normpath(koboldai_vars.custmodpth))
        return modelname
    else:
        modelname = (
            koboldai_vars.model if koboldai_vars.model is not None else "Read Only"
        )
        return modelname


def get_config_filename(model_name=None):
    if model_name:
        return f"settings/{model_name.replace('/', '_')}.settings"
    elif args.configname:
        return f"settings/{args.configname.replace('/', '_')}.settings"
    elif koboldai_vars.configname != "":
        return f"settings/{koboldai_vars.configname.replace('/', '_')}.settings"
    else:
        logger.warning(f"Empty configfile name sent back. Defaulting to ReadOnly")
        return f"settings/ReadOnly.settings"


def get_model_size(model_name):
    if "30B" in model_name:
        return "30B"
    elif "20B" in model_name:
        return "20B"
    elif "13B" in model_name:
        return "13B"
    elif "6B" in model_name.replace("6.7B", "6B"):
        return "6B"
    elif "2.7B" in model_name:
        return "2.7B"
    elif "1.3B" in model_name:
        return "1.3B"


def loadsettings():
    if path.exists("settings/" + getmodelname().replace("/", "_") + ".v2_settings"):
        with open(
            "settings/" + getmodelname().replace("/", "_") + ".v2_settings", "r"
        ) as file:
            getattr(koboldai_vars, "_model_settings").from_json(file.read())


def model_info():
    if model_config is not None:
        if isinstance(model_config, dict):
            if "model_type" in model_config:
                model_type = str(model_config["model_type"])
            elif koboldai_vars.mode[:4] == "gpt2":
                model_type = "gpt2"
            else:
                model_type = "Unknown"
        else:
            model_type = str(model_config.model_type)
        return {
            "Model Type": model_type,
            "Model Size": get_model_size(koboldai_vars.model),
            "Model Name": koboldai_vars.model.replace("_", "/"),
        }
    else:
        return {
            "Model Type": "Read Only",
            "Model Size": "0",
            "Model Name": koboldai_vars.model.replace("_", "/"),
        }


def loadmodelsettings():
    try:
        js = json.loads(str(model.model_config).partition(" ")[2])
    except Exception as e:
        try:
            try:
                js = json.load(open(koboldai_vars.custmodpth + "/config.json", "r"))
            except Exception as e:
                js = json.load(
                    open(
                        koboldai_vars.custmodpth.replace("/", "_") + "/config.json", "r"
                    )
                )
        except Exception as e:
            js = {}
    koboldai_vars.default_preset = koboldai_settings.default_preset
    if koboldai_vars.model_type == "xglm" or js.get("compat", "j") == "fairseq_lm":
        koboldai_vars.newlinemode = "s"  # Default to </s> newline mode if using XGLM
    if koboldai_vars.model_type == "opt" or koboldai_vars.model_type == "bloom":
        koboldai_vars.newlinemode = "ns"  # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them
    koboldai_vars.modelconfig = js
    if "badwordsids" in js:
        koboldai_vars.badwordsids = js["badwordsids"]
    if "nobreakmodel" in js:
        koboldai_vars.nobreakmodel = js["nobreakmodel"]
    if "sampler_order" in js:
        sampler_order = js["sampler_order"]
        if len(sampler_order) < 7:
            sampler_order = [6] + sampler_order
        koboldai_vars.sampler_order = sampler_order
    if "temp" in js:
        koboldai_vars.temp = js["temp"]
        koboldai_vars.default_preset["temp"] = js["temp"]
    if "top_p" in js:
        koboldai_vars.top_p = js["top_p"]
        koboldai_vars.default_preset["top_p"] = js["top_p"]
    if "top_k" in js:
        koboldai_vars.top_k = js["top_k"]
        koboldai_vars.default_preset["top_k"] = js["top_k"]
    if "tfs" in js:
        koboldai_vars.tfs = js["tfs"]
        koboldai_vars.default_preset["tfs"] = js["tfs"]
    if "typical" in js:
        koboldai_vars.typical = js["typical"]
        koboldai_vars.default_preset["typical"] = js["typical"]
    if "top_a" in js:
        koboldai_vars.top_a = js["top_a"]
        koboldai_vars.default_preset["top_a"] = js["top_a"]
    if "rep_pen" in js:
        koboldai_vars.rep_pen = js["rep_pen"]
        koboldai_vars.default_preset["rep_pen"] = js["rep_pen"]
    if "rep_pen_slope" in js:
        koboldai_vars.rep_pen_slope = js["rep_pen_slope"]
        koboldai_vars.default_preset["rep_pen_slope"] = js["rep_pen_slope"]
    if "rep_pen_range" in js:
        koboldai_vars.rep_pen_range = js["rep_pen_range"]
        koboldai_vars.default_preset["rep_pen_range"] = js["rep_pen_range"]
    if "adventure" in js:
        koboldai_vars.adventure = js["adventure"]
    if "chatmode" in js:
        koboldai_vars.chatmode = js["chatmode"]
    if "dynamicscan" in js:
        koboldai_vars.dynamicscan = js["dynamicscan"]
    if "formatoptns" in js:
        for setting in [
            "frmttriminc",
            "frmtrmblln",
            "frmtrmspch",
            "frmtadsnsp",
            "singleline",
        ]:
            if setting in js["formatoptns"]:
                setattr(koboldai_vars, setting, js["formatoptns"][setting])

    if "newlinemode" in js:
        koboldai_vars.newlinemode = js["newlinemode"]
    if "antemplate" in js:
        koboldai_vars.setauthornotetemplate = js["antemplate"]
        if not koboldai_vars.gamestarted:
            koboldai_vars.authornotetemplate = koboldai_vars.setauthornotetemplate


def final_startup():
    # Prevent tokenizer from taking extra time the first time it's used
    def __preempt_tokenizer():
        if "tokenizer" not in globals():
            return
        utils.decodenewlines(tokenizer.decode([25678, 559]))
        tokenizer.encode(utils.encodenewlines("eunoia"))

    tpool.execute(__preempt_tokenizer)

    # Load soft prompt specified by the settings file, if applicable
    # if(path.exists("settings/" + getmodelname().replace('/', '_') + ".v2_settings")):
    #     file = open("settings/" + getmodelname().replace('/', '_') + ".v2_settings", "r")
    #     js   = json.load(file)
    # if(koboldai_vars.allowsp and "softprompt" in js and type(js["softprompt"]) is str and all(q not in js["softprompt"] for q in ("..", ":")) and (len(js["softprompt"]) != 0 and all(js["softprompt"][0] not in q for q in ("/", "\\")))):
    #     if valid_softprompt("softprompts/"+js["softprompt"]):
    #         spRequest(js["softprompt"])
    # else:
    #     koboldai_vars.spfilename = ""
    # file.close()

    # Precompile TPU backend if required
    if model and model.capabilties.uses_tpu:
        model.raw_generate([23403, 727, 20185], max_new=1)

    # Set the initial RNG seed
    # set_seed()


# def reset_model_settings():
#     koboldai_vars.reset_for_model_load()


def load_model(
    use_gpu=True,
    gpu_layers=None,
    disk_layers=None,
    initial_load=False,
    online_model="",
    use_breakmodel_args=False,
    breakmodel_args_default_to_cpu=False,
    url=None,
    use_8_bit=False,
):
    global model
    global tokenizer
    global model_config

    koboldai_vars.aibusy = True
    koboldai_vars.horde_share = False

    if initial_load:
        use_breakmodel_args = True

    # reset_model_settings()
    # koboldai_vars.reset_model()

    koboldai_vars.cluster_requested_models = (
        [online_model] if isinstance(online_model, str) else online_model
    )
    if koboldai_vars.cluster_requested_models == [""]:
        koboldai_vars.cluster_requested_models = []

    koboldai_vars.noai = False
    if gpu_layers is not None:
        args.breakmodel_gpulayers = gpu_layers
    elif use_breakmodel_args:
        gpu_layers = args.breakmodel_gpulayers
    if breakmodel_args_default_to_cpu and gpu_layers is None:
        gpu_layers = args.breakmodel_gpulayers = []
    if disk_layers is not None:
        args.breakmodel_disklayers = int(disk_layers)
    elif use_breakmodel_args:
        disk_layers = args.breakmodel_disklayers
    if breakmodel_args_default_to_cpu and disk_layers is None:
        disk_layers = args.breakmodel_disklayers = 0

    unload_model()

    if online_model == "":
        koboldai_vars.configname = getmodelname()
    # Let's set the GooseAI or OpenAI server URLs if that's applicable
    else:
        koboldai_vars.online_model = online_model
        # Swap OAI Server if GooseAI was selected
        if koboldai_vars.model == "GooseAI":
            koboldai_vars.oaiengines = "https://api.goose.ai/v1/engines"
            koboldai_vars.model = "OAI"
            koboldai_vars.configname = f"GooseAI_{online_model.replace('/', '_')}"
        elif koboldai_vars.model == "CLUSTER" and isinstance(online_model, list):
            if len(online_model) != 1:
                koboldai_vars.configname = koboldai_vars.model
            else:
                koboldai_vars.configname = (
                    f"{koboldai_vars.model}_{online_model[0].replace('/', '_')}"
                )
        else:
            koboldai_vars.configname = (
                f"{koboldai_vars.model}_{online_model.replace('/', '_')}"
            )

        if path.exists(get_config_filename()):
            changed = False
            with open(get_config_filename(), "r") as file:
                # Check if API key exists
                js = json.load(file)
                if "online_model" in js:
                    if js["online_model"] != online_model:
                        changed = True
                        js["online_model"] = online_model
                else:
                    changed = True
                    js["online_model"] = online_model

            if changed:
                with open(
                    "settings/{}.v2_settings".format(koboldai_vars.model), "w"
                ) as file:
                    file.write(json.dumps(js, indent=3))

        # Swap OAI Server if GooseAI was selected
        if koboldai_vars.model == "GooseAI":
            koboldai_vars.oaiengines = "https://api.goose.ai/v1/engines"
            koboldai_vars.model = "OAI"
            args.configname = "GooseAI" + "/" + online_model
        elif koboldai_vars.model != "CLUSTER":
            args.configname = koboldai_vars.model + "/" + online_model
        koboldai_vars.oaiurl = koboldai_vars.oaiengines + "/{0}/completions".format(
            online_model
        )

    # If transformers model was selected & GPU available, ask to use CPU or GPU
    if not koboldai_vars.use_colab_tpu and koboldai_vars.model not in [
        "InferKit",
        "Colab",
        "API",
        "CLUSTER",
        "OAI",
        "GooseAI",
        "ReadOnly",
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    ]:
        # loadmodelsettings()
        # loadsettings()
        logger.init("GPU support", status="Searching")
        koboldai_vars.hascuda = torch.cuda.is_available() and not args.cpu
        koboldai_vars.bmsupported = (
            (koboldai_vars.model_type != "gpt2")
            or koboldai_vars.model_type in ("gpt_neo", "gptj", "xglm", "opt")
        ) and not koboldai_vars.nobreakmodel
        if args.breakmodel is not None and args.breakmodel:
            logger.warning(
                "--breakmodel is no longer supported. Breakmodel mode is now automatically enabled when --breakmodel_gpulayers is used (see --help for details)."
            )
        if args.breakmodel_layers is not None:
            logger.warning(
                "--breakmodel_layers is deprecated. Use --breakmodel_gpulayers instead (see --help for details)."
            )
        if (
            args.model
            and koboldai_vars.bmsupported
            and not args.breakmodel_gpulayers
            and not args.breakmodel_layers
            and (not args.breakmodel_disklayers)
        ):
            logger.warning(
                "Model launched without the --breakmodel_gpulayers argument, defaulting to GPU only mode."
            )
            koboldai_vars.bmsupported = False
        if not koboldai_vars.bmsupported and (
            args.breakmodel_gpulayers is not None
            or args.breakmodel_layers is not None
            or args.breakmodel_disklayers is not None
        ):
            logger.warning(
                "This model does not support hybrid generation. --breakmodel_gpulayers will be ignored."
            )
        if koboldai_vars.hascuda:
            logger.init_ok("GPU support", status="Found")
        else:
            logger.init_warn("GPU support", status="Not Found")

        if args.cpu:
            koboldai_vars.usegpu = False
            gpu_layers = None
            disk_layers = None
            koboldai_vars.breakmodel = False
        elif koboldai_vars.hascuda:
            if koboldai_vars.bmsupported:
                koboldai_vars.usegpu = False
                koboldai_vars.breakmodel = True
            else:
                koboldai_vars.breakmodel = False
                koboldai_vars.usegpu = use_gpu
    else:
        koboldai_vars.default_preset = koboldai_settings.default_preset

    # Ask for API key if InferKit was selected
    if koboldai_vars.model == "InferKit":
        koboldai_vars.apikey = koboldai_vars.oaiapikey

    # Swap OAI Server if GooseAI was selected
    if koboldai_vars.model == "GooseAI":
        koboldai_vars.oaiengines = "https://api.goose.ai/v1/engines"
        koboldai_vars.model = "OAI"
        koboldai_vars.configname = "GooseAI"

    # Ask for API key if OpenAI was selected
    if koboldai_vars.model == "OAI" and not koboldai_vars.configname:
        koboldai_vars.configname = "OAI"

    if koboldai_vars.model == "ReadOnly":
        koboldai_vars.noai = True

    # TODO: InferKit
    if koboldai_vars.model == "ReadOnly" or koboldai_vars.noai:
        pass
    elif koboldai_vars.model in ["Colab", "API", "CLUSTER", "OAI"]:
        koboldai_vars.colaburl = url or koboldai_vars.colaburl
        koboldai_vars.usegpu = False
        koboldai_vars.breakmodel = False

        if koboldai_vars.model == "Colab":
            from modeling.inference_models.basic_api import BasicAPIInferenceModel

            model = BasicAPIInferenceModel()
        elif koboldai_vars.model == "API":
            from modeling.inference_models.api import APIInferenceModel

            model = APIInferenceModel(koboldai_vars.colaburl.replace("/request", ""))
        elif koboldai_vars.model == "CLUSTER":
            from modeling.inference_models.horde import HordeInferenceModel

            model = HordeInferenceModel()
        elif koboldai_vars.model == "OAI":
            from modeling.inference_models.openai import OpenAIAPIInferenceModel

            model = OpenAIAPIInferenceModel()

        model.load(initial_load=initial_load)
    # TODO: This check sucks, make a model object or somethign
    elif "rwkv" in koboldai_vars.model:
        if koboldai_vars.use_colab_tpu:
            raise RuntimeError("RWKV is not supported on the TPU.")
        from modeling.inference_models.rwkv import RWKVInferenceModel

        model = RWKVInferenceModel(koboldai_vars.model)
        model.load()
    elif not koboldai_vars.use_colab_tpu and not koboldai_vars.noai:
        # HF Torch
        logger.init("Transformers", status="Starting")
        for m in ("GPTJModel", "XGLMModel"):
            try:
                globals()[m] = getattr(__import__("transformers"), m)
            except:
                pass

        from modeling.inference_models.generic_hf_torch import (
            GenericHFTorchInferenceModel,
        )

        model = GenericHFTorchInferenceModel(
            koboldai_vars.model, lazy_load=koboldai_vars.lazy_load, low_mem=args.lowmem
        )

        model.load(
            save_model=True,
            initial_load=initial_load,
        )
        logger.info(f"Pipeline created: {koboldai_vars.model}")
    else:
        # TPU
        from modeling.inference_models.hf_mtj import HFMTJInferenceModel

        model = HFMTJInferenceModel(koboldai_vars.model)
        model.load(
            save_model=not (args.colab or args.cacheonly) or args.savemodel,
            initial_load=initial_load,
        )

    # TODO: Convert everywhere to use model.tokenizer
    if model:
        tokenizer = model.tokenizer

    loadmodelsettings()
    loadsettings()

    # lua_startup()
    # Load scripts
    # load_lua_scripts()

    final_startup()

    # if not koboldai_vars.gamestarted:
    #     setStartState()
    #     sendsettings()
    #     refresh_settings()

    # Saving the tokenizer to the KoboldStoryRegister class so we can do token counting on the story data
    if "tokenizer" in [x for x in globals()]:
        koboldai_vars.tokenizer = tokenizer

    # Let's load the presets
    preset_same_model = {}
    preset_same_class_size = {}
    preset_same_class = {}
    preset_others = {}
    model_info_data = model_info()

    for file in os.listdir("./presets"):
        if file[-8:] == ".presets":
            with open("./presets/{}".format(file)) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            for preset in data:
                if preset["Model Name"] == koboldai_vars.model:
                    preset_same_model[preset["preset"]] = preset
                    preset_same_model[preset["preset"]]["Match"] = "Recommended"
                elif (
                    not (
                        preset["preset"] in preset_same_model
                        and preset_same_model[preset["preset"]]["Match"]
                        == "Recommended"
                    )
                    and model_info_data["Model Type"] == preset["Model Type"]
                    and model_info_data["Model Size"] == preset["Model Size"]
                ):
                    preset_same_class_size[preset["preset"]] = preset
                    preset_same_class_size[preset["preset"]]["Match"] = "Recommended"
                elif (
                    not (
                        preset["preset"] in preset_same_model
                        and preset_same_model[preset["preset"]]["Match"]
                        == "Recommended"
                    )
                    and not (
                        (
                            preset["preset"] in preset_same_class_size
                            and preset_same_class_size[preset["preset"]]["Match"]
                            == "Recommended"
                        )
                    )
                    and model_info_data["Model Type"] == preset["Model Type"]
                ):
                    preset_same_class[preset["preset"]] = preset
                    preset_same_class[preset["preset"]]["Match"] = "Same Class"
                elif (
                    preset["preset"] not in preset_same_model
                    and preset["preset"] not in preset_same_class_size
                    and preset["preset"] not in preset_same_class
                ):
                    preset_others[preset["preset"]] = preset
                    preset_others[preset["preset"]]["Match"] = "Other"

    # Combine it all
    presets = preset_same_model
    for item in preset_same_class_size:
        if item not in presets:
            presets[item] = preset_same_class_size[item]
    for item in preset_same_class:
        if item not in presets:
            presets[item] = preset_same_class[item]
    for item in preset_others:
        if item not in presets:
            presets[item] = preset_others[item]

    presets["Default"] = koboldai_vars.default_preset

    koboldai_vars.uid_presets = presets
    # We want our data to be a 2 deep dict. Top level is "Recommended", "Same Class", "Model 1", "Model 2", etc
    # Next layer is "Official", "Custom"
    # Then the preset name

    to_use = OrderedDict()

    to_use["Recommended"] = {
        "Official": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Recommended"
            and presets[x]["Preset Category"] == "Official"
        ],
        "Custom": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Recommended"
            and presets[x]["Preset Category"] == "Custom"
        ],
    }
    to_use["Same Class"] = {
        "Official": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Same Class"
            and presets[x]["Preset Category"] == "Official"
        ],
        "Custom": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Same Class"
            and presets[x]["Preset Category"] == "Custom"
        ],
    }
    to_use["Other"] = {
        "Official": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Other"
            and presets[x]["Preset Category"] == "Official"
        ],
        "Custom": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Other"
            and presets[x]["Preset Category"] == "Custom"
        ],
    }
    koboldai_vars.presets = to_use

    koboldai_vars.aibusy = False
    if not os.path.exists("./softprompts"):
        os.mkdir("./softprompts")

    return model
    # koboldai_vars.splist = [[f, get_softprompt_desc(os.path.join("./softprompts", f),None,True)] for f in os.listdir("./softprompts") if os.path.isfile(os.path.join("./softprompts", f)) and valid_softprompt(os.path.join("./softprompts", f))]
    # if initial_load and koboldai_vars.cloudflare_link != "":
    #     print(format(colors.GREEN) + "KoboldAI has finished loading and is available at the following link for UI 1: " + koboldai_vars.cloudflare_link + format(colors.END))
    #     print(format(colors.GREEN) + "KoboldAI has finished loading and is available at the following link for UI 2: " + koboldai_vars.cloudflare_link + "/new_ui" + format(colors.END))


def apiactionsubmit_generate(txt, minimum, maximum):
    koboldai_vars.generated_tkns = 0

    if not koboldai_vars.quiet:
        logger.debug(f"Prompt Min:{minimum}, Max:{maximum}")
        logger.prompt(
            utils.decodenewlines(tokenizer.decode(txt))
            .encode("unicode_escape")
            .decode("utf-8")
        )

    # Clear CUDA cache if using GPU
    if koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    _genout, already_generated = tpool.execute(model.core_generate, txt, set())

    genout = [
        utils.applyoutputformatting(
            utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]))
        )
        for tokens in _genout
    ]

    # Clear CUDA cache again if using GPU
    if koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel):
        del _genout
        gc.collect()
        torch.cuda.empty_cache()

    return genout


def applyinputformatting(txt):
    # Add sentence spacing
    if koboldai_vars.frmtadsnsp and not koboldai_vars.chatmode:
        txt = utils.addsentencespacing(txt, koboldai_vars)

    return txt


def apiactionsubmit(
    data,
    use_memory=False,
    use_world_info=False,
    use_story=False,
    use_authors_note=False,
):
    if not model or not model.capabilties.api_host:
        raise NotImplementedError(
            f"API generation isn't allowed on model '{koboldai_vars.model}'"
        )

    data = applyinputformatting(data)

    if koboldai_vars.memory != "" and koboldai_vars.memory[-1] != "\n":
        mem = koboldai_vars.memory + "\n"
    else:
        mem = koboldai_vars.memory

    if use_authors_note and koboldai_vars.authornote != "":
        anotetxt = ("\n" + koboldai_vars.authornotetemplate + "\n").replace(
            "<|>", koboldai_vars.authornote
        )
    else:
        anotetxt = ""

    MIN_STORY_TOKENS = 8
    story_tokens = []
    mem_tokens = []
    wi_tokens = []

    story_budget = (
        lambda: koboldai_vars.max_length
        - koboldai_vars.sp_length
        - koboldai_vars.genamt
        - len(tokenizer._koboldai_header)
        - len(story_tokens)
        - len(mem_tokens)
        - len(wi_tokens)
    )
    budget = lambda: story_budget() + MIN_STORY_TOKENS
    if budget() < 0:
        return {
            "detail": {
                "msg": f"Your Max Tokens setting is too low for your current soft prompt and tokenizer to handle. It needs to be at least {koboldai_vars.max_length - budget()}.",
                "type": "token_overflow",
            }
        }
        # abort(Response(json.dumps({"detail": {
        #     "msg": f"Your Max Tokens setting is too low for your current soft prompt and tokenizer to handle. It needs to be at least {koboldai_vars.max_length - budget()}.",
        #     "type": "token_overflow",
        # }}), mimetype="application/json", status=500))

    if use_memory:
        mem_tokens = tokenizer.encode(utils.encodenewlines(mem))[-budget() :]

    if use_world_info:
        # world_info, _ = checkworldinfo(data, force_use_txt=True, scan_story=use_story)
        world_info = koboldai_vars.worldinfo_v2.get_used_wi()
        wi_tokens = tokenizer.encode(utils.encodenewlines(world_info))[-budget() :]

    if use_story:
        if koboldai_vars.useprompt:
            story_tokens = tokenizer.encode(utils.encodenewlines(koboldai_vars.prompt))[
                -budget() :
            ]

    story_tokens = (
        tokenizer.encode(utils.encodenewlines(data))[-story_budget() :] + story_tokens
    )

    if use_story:
        for i, action in enumerate(reversed(koboldai_vars.actions.values())):
            if story_budget() <= 0:
                assert story_budget() == 0
                break
            story_tokens = (
                tokenizer.encode(utils.encodenewlines(action))[-story_budget() :]
                + story_tokens
            )
            if i == koboldai_vars.andepth - 1:
                story_tokens = (
                    tokenizer.encode(utils.encodenewlines(anotetxt))[-story_budget() :]
                    + story_tokens
                )
        if not koboldai_vars.useprompt:
            story_tokens = (
                tokenizer.encode(utils.encodenewlines(koboldai_vars.prompt))[
                    -budget() :
                ]
                + story_tokens
            )

    tokens = tokenizer._koboldai_header + mem_tokens + wi_tokens + story_tokens
    assert story_budget() >= 0
    minimum = len(tokens) + 1
    maximum = len(tokens) + koboldai_vars.genamt

    startTime = time.time()

    if not koboldai_vars.use_colab_tpu and koboldai_vars.model not in [
        "Colab",
        "API",
        "CLUSTER",
        "OAI",
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    ]:
        genout = apiactionsubmit_generate(tokens, minimum, maximum)

    endTime = time.time()
    howMuchTime = endTime - startTime
    print(str(howMuchTime) + " sec")
    return genout


def permutation_validator(lst: list):
    if any(not isinstance(e, int) for e in lst):
        return
    if min(lst) != 0 or max(lst) != len(lst) - 1 or len(set(lst)) != len(lst):
        raise ValidationError(
            "Must be a permutation of the first N non-negative integers, where N is the length of this array"
        )
    return True


class KoboldSchema(Schema):
    pass


class SamplerSettingsSchema(KoboldSchema):
    rep_pen: Optional[float] = fields.Float(
        validate=validate.Range(min=1),
        metadata={"description": "Base repetition penalty value."},
    )
    rep_pen_range: Optional[int] = fields.Integer(
        validate=validate.Range(min=0),
        metadata={"description": "Repetition penalty range."},
    )
    rep_pen_slope: Optional[float] = fields.Float(
        validate=validate.Range(min=0),
        metadata={"description": "Repetition penalty slope."},
    )
    top_k: Optional[int] = fields.Integer(
        validate=validate.Range(min=0),
        metadata={"description": "Top-k sampling value."},
    )
    top_a: Optional[float] = fields.Float(
        validate=validate.Range(min=0),
        metadata={"description": "Top-a sampling value."},
    )
    top_p: Optional[float] = fields.Float(
        validate=validate.Range(min=0, max=1),
        metadata={"description": "Top-p sampling value."},
    )
    tfs: Optional[float] = fields.Float(
        validate=validate.Range(min=0, max=1),
        metadata={"description": "Tail free sampling value."},
    )
    typical: Optional[float] = fields.Float(
        validate=validate.Range(min=0, max=1),
        metadata={"description": "Typical sampling value."},
    )
    temperature: Optional[float] = fields.Float(
        validate=validate.Range(min=0, min_inclusive=False),
        metadata={"description": "Temperature value."},
    )


class GenerationInputSchema(SamplerSettingsSchema):
    prompt: str = fields.String(
        required=True, metadata={"description": "This is the submission."}
    )
    use_memory: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the memory from the KoboldAI GUI when generating text."
        },
    )
    use_story: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the story from the KoboldAI GUI when generating text."
        },
    )
    use_authors_note: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the author's note from the KoboldAI GUI when generating text. This has no effect unless `use_story` is also enabled."
        },
    )
    use_world_info: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the world info from the KoboldAI GUI when generating text."
        },
    )
    use_userscripts: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the userscripts from the KoboldAI GUI when generating text."
        },
    )
    soft_prompt: Optional[str]
    max_length: int = fields.Integer(
        validate=validate.Range(min=1, max=512),
        metadata={"description": "Number of tokens to generate."},
    )
    max_context_length: int = fields.Integer(
        validate=validate.Range(min=1),
        metadata={"description": "Maximum number of tokens to send to the model."},
    )
    n: int = fields.Integer(
        validate=validate.Range(min=1, max=5),
        metadata={"description": "Number of outputs to generate."},
    )
    disable_output_formatting: bool = fields.Boolean(
        load_default=True,
        metadata={
            "description": "When enabled, all output formatting options default to `false` instead of the value in the KoboldAI GUI."
        },
    )
    frmttriminc: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Output formatting option. When enabled, removes some characters from the end of the output such that the output doesn't end in the middle of a sentence. If the output is less than one sentence long, does nothing.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    frmtrmblln: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Output formatting option. When enabled, replaces all occurrences of two or more consecutive newlines in the output with one newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    frmtrmspch: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Output formatting option. When enabled, removes `#/@%{}+=~|\^<>` from the output.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    singleline: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Output formatting option. When enabled, removes everything after the first line of the output, including the newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    disable_input_formatting: bool = fields.Boolean(
        load_default=True,
        metadata={
            "description": "When enabled, all input formatting options default to `false` instead of the value in the KoboldAI GUI"
        },
    )
    frmtadsnsp: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Input formatting option. When enabled, adds a leading space to your input if there is no trailing whitespace at the end of the previous action.\n\nIf `disable_input_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    quiet: Optional[bool] = fields.Boolean(
        metadata={
            "description": "When enabled, Generated output will not be displayed in the console."
        }
    )
    sampler_order: Optional[List[int]] = fields.List(
        fields.Integer(),
        validate=[validate.Length(min=6), permutation_validator],
        metadata={
            "description": "Sampler order to be used. If N is the length of this array, then N must be greater than or equal to 6 and the array must be a permutation of the first N non-negative integers."
        },
    )
    sampler_seed: Optional[int] = fields.Integer(
        validate=validate.Range(min=0, max=2**64 - 1),
        metadata={
            "description": "RNG seed to use for sampling. If not specified, the global RNG will be used."
        },
    )
    sampler_full_determinism: Optional[bool] = fields.Boolean(
        metadata={
            "description": "If enabled, the generated text will always be the same as long as you use the same RNG seed, input and settings. If disabled, only the *sequence* of generated texts that you get when repeatedly generating text will be the same given the same RNG seed, input and settings."
        }
    )
    stop_sequence: Optional[List[str]] = fields.List(
        fields.String(),
        metadata={
            "description": "An array of string sequences where the API will stop generating further tokens. The returned text WILL contain the stop sequence."
        },
        validate=[validate.Length(max=10)],
    )


def _generate_text(body: GenerationInputSchema):
    if hasattr(body, "sampler_seed"):
        # If a seed was specified, we need to save the global RNG state so we
        # can restore it later
        old_seed = koboldai_vars.seed
        # old_rng_state = tpu_mtj_backend.get_rng_state() if koboldai_vars.use_colab_tpu else torch.get_rng_state()
        old_rng_state = torch.get_rng_state()
        koboldai_vars.seed = body.sampler_seed
        # We should try to use a previously saved RNG state with the same seed
        if body.sampler_seed in koboldai_vars.rng_states:
            torch.set_rng_state(koboldai_vars.rng_states[body.sampler_seed])
            # if koboldai_vars.use_colab_tpu:
            #     tpu_mtj_backend.set_rng_state(koboldai_vars.rng_states[body.sampler_seed])
            # else:
            #     torch.set_rng_state(koboldai_vars.rng_states[body.sampler_seed])
        else:
            torch.manual_seed(body.sampler_seed)
            # if koboldai_vars.use_colab_tpu:
            #     tpu_mtj_backend.set_rng_state(tpu_mtj_backend.new_rng_state(body.sampler_seed))
            # else:
            #     torch.manual_seed(body.sampler_seed)
        # koboldai_vars.rng_states[body.sampler_seed] = tpu_mtj_backend.get_rng_state() if koboldai_vars.use_colab_tpu else torch.get_rng_state()
        koboldai_vars.rng_states[body.sampler_seed] = torch.get_rng_state()
    if hasattr(body, "sampler_order"):
        if len(body.sampler_order) < 7:
            body.sampler_order = [6] + body.sampler_order
    # This maps each property of the setting to use when sending the generate idempotently
    # To the object which typically contains it's value
    # This allows to set the property only for the API generation, and then revert the setting
    # To what it was before.
    mapping = {
        "disable_input_formatting": ("koboldai_vars", "disable_input_formatting", None),
        "disable_output_formatting": (
            "koboldai_vars",
            "disable_output_formatting",
            None,
        ),
        "rep_pen": ("koboldai_vars", "rep_pen", None),
        "rep_pen_range": ("koboldai_vars", "rep_pen_range", None),
        "rep_pen_slope": ("koboldai_vars", "rep_pen_slope", None),
        "top_k": ("koboldai_vars", "top_k", None),
        "top_a": ("koboldai_vars", "top_a", None),
        "top_p": ("koboldai_vars", "top_p", None),
        "tfs": ("koboldai_vars", "tfs", None),
        "typical": ("koboldai_vars", "typical", None),
        "temperature": ("koboldai_vars", "temp", None),
        "frmtadsnsp": ("koboldai_vars", "frmtadsnsp", "input"),
        "frmttriminc": ("koboldai_vars", "frmttriminc", "output"),
        "frmtrmblln": ("koboldai_vars", "frmtrmblln", "output"),
        "frmtrmspch": ("koboldai_vars", "frmtrmspch", "output"),
        "singleline": ("koboldai_vars", "singleline", "output"),
        "max_length": ("koboldai_vars", "genamt", None),
        "max_context_length": ("koboldai_vars", "max_length", None),
        "n": ("koboldai_vars", "numseqs", None),
        "quiet": ("koboldai_vars", "quiet", None),
        "sampler_order": ("koboldai_vars", "sampler_order", None),
        "sampler_full_determinism": ("koboldai_vars", "full_determinism", None),
        "stop_sequence": ("koboldai_vars", "stop_sequence", None),
    }
    saved_settings = {}
    disable_set_aibusy = koboldai_vars.disable_set_aibusy
    koboldai_vars.disable_set_aibusy = True
    _standalone = koboldai_vars.standalone
    koboldai_vars.standalone = True
    show_probs = koboldai_vars.show_probs
    koboldai_vars.show_probs = False
    output_streaming = koboldai_vars.output_streaming
    koboldai_vars.output_streaming = False
    for key, entry in mapping.items():
        obj = {"koboldai_vars": koboldai_vars}[entry[0]]
        if (
            entry[2] == "input"
            and koboldai_vars.disable_input_formatting
            and not hasattr(body, key)
        ):
            setattr(body, key, False)
        if (
            entry[2] == "output"
            and koboldai_vars.disable_output_formatting
            and not hasattr(body, key)
        ):
            setattr(body, key, False)
        if getattr(body, key, None) is not None:
            if entry[1].startswith("@"):
                saved_settings[key] = obj[entry[1][1:]]
                obj[entry[1][1:]] = getattr(body, key)
            else:
                saved_settings[key] = getattr(obj, entry[1])
                setattr(obj, entry[1], getattr(body, key))
    try:
        if koboldai_vars.allowsp and getattr(body, "soft_prompt", None) is not None:
            if any(q in body.soft_prompt for q in ("/", "\\")):
                raise RuntimeError
            old_spfilename = koboldai_vars.spfilename
            # spRequest(body.soft_prompt.strip())
        genout = apiactionsubmit(
            body.prompt,
            use_memory=body.use_memory,
            use_story=body.use_story,
            use_world_info=body.use_world_info,
            use_authors_note=body.use_authors_note,
        )
        output = {"results": [{"text": txt} for txt in genout]}
    finally:
        for key in saved_settings:
            entry = mapping[key]
            obj = {"koboldai_vars": koboldai_vars}[entry[0]]
            if getattr(body, key, None) is not None:
                if entry[1].startswith("@"):
                    if obj[entry[1][1:]] == getattr(body, key):
                        obj[entry[1][1:]] = saved_settings[key]
                else:
                    if getattr(obj, entry[1]) == getattr(body, key):
                        setattr(obj, entry[1], saved_settings[key])
        koboldai_vars.disable_set_aibusy = disable_set_aibusy
        koboldai_vars.standalone = _standalone
        koboldai_vars.show_probs = show_probs
        koboldai_vars.output_streaming = output_streaming
        # if koboldai_vars.allowsp and getattr(body, "soft_prompt", None) is not None:
        #     spRequest(old_spfilename)
        if hasattr(body, "sampler_seed"):
            koboldai_vars.seed = old_seed
            torch.set_rng_state(old_rng_state)
            # if koboldai_vars.use_colab_tpu:
            #     tpu_mtj_backend.set_rng_state(old_rng_state)
            # else:
            #     torch.set_rng_state(old_rng_state)
    return output
