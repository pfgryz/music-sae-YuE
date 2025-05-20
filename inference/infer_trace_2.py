# region Load local models
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcodec_mini_infer"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcodec_mini_infer", "descriptaudiocodec"))
# endregion

from dataclasses import dataclass

import hydra
from nnsight import LanguageModel
import torch
from transformers import AutoModelForCausalLM, LogitsProcessorList

from common import initialize_seed, split_lyrics, BlockTokenRangeProcessor
from yue import YuEInferenceConfig, YuEProcessorConfig, YuEProcessor


@dataclass
class CollectScriptConfig:
    device: str = "cuda"
    model: str = "7B-anneal-en-cot"
    layer: int = 13
    min_new_tokens: int = 3000
    max_new_tokens: int = 3000
    seed: int = 42
    inference: YuEInferenceConfig = YuEInferenceConfig()
    processor: YuEProcessorConfig = YuEProcessorConfig()


@hydra.main(version_base=None, config_path="../conf/yue/", config_name="config")
def main(args: CollectScriptConfig):
    initialize_seed(args.seed)
    device = torch.device(args.device)

    processor = YuEProcessor(device, args.processor)

    model = AutoModelForCausalLM.from_pretrained(
        f"m-a-p/YuE-s1-{args.model}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = LanguageModel(model, input_names=["input_ids"])
    model.to(device)
    model.eval()

    # region INPUT @TODO: replace with loading from datasets
    def load_audio_mono(filepath, sampling_rate=16000):
        import torchaudio
        from torchaudio.transforms import Resample

        audio, sr = torchaudio.load(filepath)
        # Convert to mono
        audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample if needed
        if sr != sampling_rate:
            resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
            audio = resampler(audio)
        return audio

    GENRES_PATH = "../examples/genres/g0.txt"
    LYRICS_PATH = "../examples/lyrics.txt"
    MUSIC_PATH = "placeholder.mp3"

    audio_prompt = load_audio_mono(MUSIC_PATH)
    with open(GENRES_PATH) as f:
        genres = f.read().strip()
    with open(LYRICS_PATH) as f:
        lyrics = split_lyrics(f.read())
    # endregion

    # region CHOOSE LAYER
    layer = model.model.layers[args.layer]

    with torch.no_grad():
        inputs = processor.process(genres, lyrics, audio_prompt)

        with model.trace(
            inputs=inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=True,
            top_p=args.inference.top_p,
            temperature=args.inference.temperature,
            repetition_penalty=args.inference.repetition_penalty,
            eos_token_id=processor.eoa,
            pad_token_id=processor.eoa,
            logits_processor=LogitsProcessorList(
                [BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]
            ),
            guidance_scale=args.inference.guidance_scale,
        ):
            trace = layer.output[0].save()

        print(trace.shape)
        # return trace
        # @TODO: in this place take the trace and save activations

        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
