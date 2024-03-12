import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Load the configuration
config = XttsConfig()
config.load_json("config.json")

# Initialize the model from the configuration
model = Xtts.init_from_config(config)

# Load the checkpoint
model.load_checkpoint(config, checkpoint_dir="model", eval=True)
model.cuda(device)



outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="sample.wav",
    gpt_cond_len=3,
    language="en"
)
