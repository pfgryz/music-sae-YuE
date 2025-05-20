import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcodec_mini_infer"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcodec_mini_infer", "descriptaudiocodec"))
import argparse
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import BatchEncoding, AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList  # noqa: F401
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator
from mmtokenizer import MMSentencePieceTokenizer
from nnsight import LanguageModel
from dataclasses import dataclass
from models.soundstream_hubert_new import SoundStream  # noqa: F401


from common import initialize_seed
from stage_one_utils import split_lyrics, BlockTokenRangeProcessor

# region Configuration

parser = argparse.ArgumentParser()
# Model Configuration:
parser.add_argument(
    "--stage1_model",
    type=str,
    default="m-a-p/YuE-s1-7B-anneal-en-cot",
    help="The model checkpoint path or identifier for the Stage 1 model.",
)
parser.add_argument(
    "--stage2_model",
    type=str,
    default="m-a-p/YuE-s2-1B-general",
    help="The model checkpoint path or identifier for the Stage 2 model.",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=3000,
    help="The maximum number of new tokens to generate in one pass during text generation.",
)
parser.add_argument(
    "--repetition_penalty",
    type=float,
    default=1.1,
    help="repetition_penalty ranges from 1.0 to 2.0 (or higher in some cases). It controls the diversity and coherence of the audio tokens generated. The higher the value, the greater the discouragement of repetition. Setting value to 1.0 means no penalty.",
)
parser.add_argument(
    "--run_n_segments", type=int, default=2, help="The number of segments to process during the generation."
)
parser.add_argument("--stage2_batch_size", type=int, default=4, help="The batch size used in Stage 2 inference.")
# Prompt
parser.add_argument(
    "--genre_txt",
    type=str,
    required=True,
    help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.",
)
parser.add_argument(
    "--lyrics_txt",
    type=str,
    required=True,
    help="The file path to a text file containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.",
)

# Output
parser.add_argument(
    "--output_dir", type=str, default="./output", help="The directory where generated outputs will be saved."
)
parser.add_argument("--output_file", type=str, default="", help="Name of output file.")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--seed", type=int, default=42, help="An integer value to reproduce generation.")
# Config for xcodec and upsampler
parser.add_argument(
    "--basic_model_config",
    default="./xcodec_mini_infer/final_ckpt/config.yaml",
    help="YAML files for xcodec configurations.",
)
parser.add_argument(
    "--resume_path", default="./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth", help="Path to the xcodec checkpoint."
)
parser.add_argument(
    "--config_path", type=str, default="./xcodec_mini_infer/decoders/config.yaml", help="Path to Vocos config file."
)
parser.add_argument(
    "--vocal_decoder_path",
    type=str,
    default="./xcodec_mini_infer/decoders/decoder_131000.pth",
    help="Path to Vocos decoder weights.",
)
parser.add_argument(
    "--inst_decoder_path",
    type=str,
    default="./xcodec_mini_infer/decoders/decoder_151000.pth",
    help="Path to Vocos decoder weights.",
)
parser.add_argument("-r", "--rescale", action="store_true", help="Rescale output to avoid clipping.")

# Ablation
parser.add_argument("--ablate", action="store_true", help="Run ablation")
parser.add_argument("--ablation-layer", type=int, default=0, help="Ablation layer")

# endregion

# region Configuration validation

args = parser.parse_args()

if not args.ablate and args.ablation_layer:
    raise RuntimeError("Missing ablation layer in ablate mode")

# endregion

# region Outputs paths and config


def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


# endregion

# region Init
initialize_seed(args.seed)
cuda_idx = args.cuda_idx
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
# endregion


# region Processor impl
@dataclass
class YuEProcessConfig:
    tokenizer_model_path: str = "./mm_tokenizer_v0.2_hf/tokenizer.model"
    codec_model_config_path: str = "./xcodec_mini_infer/final_ckpt/config.yaml"
    codec_model_resume_path: str = "./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"


class YuEProcessor:
    def __init__(self, device, config: YuEProcessConfig = None):
        if config is None:
            config = YuEProcessConfig()

        codec_model_config = OmegaConf.load(config.codec_model_config_path)
        codec_parameter_dict = torch.load(config.codec_model_resume_path, map_location="cpu", weights_only=False)

        self._device = device
        self._tokenizer = MMSentencePieceTokenizer(config.tokenizer_model_path)
        self._codectool = CodecManipulator("xcodec", 0, 1)

        self._codec_model = SoundStream(**codec_model_config.generator.config).to(device)
        self._codec_model.load_state_dict(codec_parameter_dict["codec_model"])
        self._codec_model.to(device)
        self._codec_model.eval()

        self._sos = self._tokenizer.tokenize("[start_of_segment]")

    @property
    def eoa(self):
        return self._tokenizer.eoa

    def _encode_audio(self, audio, target_bw=0.5):
        if len(audio.shape) < 3:
            audio.unsqueeze_(0)

        with torch.no_grad():
            raw_codes = self._codec_model.encode(audio.to(device), target_bw=target_bw)

        raw_codes = raw_codes.transpose(0, 1)
        raw_codes = raw_codes.cpu().numpy().astype(np.int16)
        return raw_codes

    def process(self, genres: str, lyrics: list[str], audio):
        full_lyrics = "\n".join(lyrics)
        segment = lyrics[0]
        prompt = f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"

        raw_codes = self._encode_audio(audio, target_bw=0.5)
        code_ids = self._codectool.npy2ids(raw_codes[0])
        audio_prompt = [self._tokenizer.soa] + self._codectool.sep_ids + code_ids + [self._tokenizer.eoa]

        sentence_ids = (
            self._tokenizer.tokenize("[start_of_reference]")
            + audio_prompt
            + self._tokenizer.tokenize("[end_of_reference]")
        )
        head_id = self._tokenizer.tokenize(prompt) + sentence_ids

        prompt_ids = (
            head_id + self._sos + self._tokenizer.tokenize(segment) + [self._tokenizer.soa] + self._codectool.sep_ids
        )
        input_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)

        attention_mask = (input_ids != 0).long()
        inputs = BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

        return inputs


# endregion

# region CONFIG
top_p = 0.93
temperature = 1.0
repetition_penalty = args.repetition_penalty
guidance_scale = 1.5
max_new_tokens = args.max_new_tokens
min_new_tokens = max_new_tokens

stage1_model = args.stage1_model
# endregion

# region Processor
processor = YuEProcessor(device)
# endregion

# region Model
model = AutoModelForCausalLM.from_pretrained(
    stage1_model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
model = LanguageModel(model, input_names=["input_ids"])
model.to(device)
model.eval()
# endregion

# region Arguments
PATH_TO_MUSIC = "placeholder.mp3"
audio_prompt = load_audio_mono(PATH_TO_MUSIC)

with open(args.genre_txt) as f:
    genres = f.read().strip()
with open(args.lyrics_txt) as f:
    lyrics = split_lyrics(f.read())
# endregion


if args.ablate:
    layer = model.model.layers[args.ablation_layer]


with torch.no_grad():
    inputs = processor.process(genres, lyrics, audio_prompt)

    with model.trace(
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        eos_token_id=processor.eoa,
        pad_token_id=processor.eoa,
        logits_processor=LogitsProcessorList(
            [BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]
        ),
        guidance_scale=guidance_scale,
    ) as tracer:
        trace = layer.output[0].save()

    print(trace.shape)
    for a in trace.view(-1, 4096).detach().cpu():
        # print(a.shape)
        pass

    # @todo: return trace here

    import sys

    sys.exit(1)
