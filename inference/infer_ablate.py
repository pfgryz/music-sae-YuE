import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcodec_mini_infer"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcodec_mini_infer", "descriptaudiocodec"))
import uuid
import copy
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from einops import rearrange
from transformers import BatchEncoding, AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList  # noqa: F401
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched
from nnsight import LanguageModel

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

stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens
stage1_output_dir = os.path.join(args.output_dir, "stage1")
stage2_output_dir = stage1_output_dir.replace("stage1", "stage2")
os.makedirs(stage1_output_dir, exist_ok=True)
os.makedirs(stage2_output_dir, exist_ok=True)

# endregion

initialize_seed(args.seed)
# load tokenizer and model
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")

# region Stage One - Load Model

mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
model = AutoModelForCausalLM.from_pretrained(
    stage1_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # To enable flashattn, you have to install flash-attn
    # device_map="auto",
)
model = LanguageModel(model, input_names=["input_ids"])
# to device, if gpu is available
model.to(device)
model.eval()

# if torch.__version__ >= "2.0.0":
#     model = torch.compile(model)

codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(args.resume_path, map_location="cpu", weights_only=False)
codec_model.load_state_dict(parameter_dict["codec_model"])
codec_model.to(device)
codec_model.eval()

# endregion

# region Stage One - Eval

# Call the function and print the result
stage1_output_set = []
# Tips:
# genre tags support instrumental，genre，mood，vocal timbr and vocal gender
# all kinds of tags are needed
with open(args.genre_txt) as f:
    genres = f.read().strip()
with open(args.lyrics_txt) as f:
    lyrics = split_lyrics(f.read())
# intruction
full_lyrics = "\n".join(lyrics)
prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
prompt_texts += lyrics

print("|", full_lyrics)
print("|", prompt_texts)

random_id = uuid.uuid4()
output_seq = None
# Here is suggested decoding config
top_p = 0.93
temperature = 1.0
repetition_penalty = args.repetition_penalty
# special tokens
start_of_segment = mmtokenizer.tokenize("[start_of_segment]")
end_of_segment = mmtokenizer.tokenize("[end_of_segment]")
raw_output = None

if args.ablate:
    layer = model.model.layers[args.ablation_layer]

min_new_tokens = max_new_tokens
# Format text prompt
run_n_segments = min(args.run_n_segments + 1, len(lyrics))
for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage1 inference...")):
    section_text = p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
    guidance_scale = 1.5 if i <= 1 else 1.2

    print("gen", i, p)

    if i == 0:
        continue
    if i == 1:
        head_id = mmtokenizer.tokenize(prompt_texts[0])
        prompt_ids = (
            head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
        )
    else:
        prompt_ids = (
            end_of_segment
            + start_of_segment
            + mmtokenizer.tokenize(section_text)
            + [mmtokenizer.soa]
            + codectool.sep_ids
        )

    prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
    input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
    # Use window slicing in case output sequence exceeds the context of model
    max_context = 16384 - max_new_tokens - 1
    if input_ids.shape[-1] > max_context:
        print(
            f"Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens."
        )
        input_ids = input_ids[:, -(max_context):]
    with torch.no_grad():
        print(model)

        attention_mask = (input_ids != 0).long()
        inputs = BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

        with model.generate(
            inputs=inputs,
            # input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=mmtokenizer.eoa,
            pad_token_id=mmtokenizer.eoa,
            logits_processor=LogitsProcessorList(
                [BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]
            ),
            guidance_scale=guidance_scale,
        ) as generator:
            output_seq = model.generator.output.save()
            for _ in range(max_new_tokens):
                if args.ablate:
                    layer.output[0][:] = layer.input[:]
                model.next()

        print(output_seq.shape)
        if output_seq[0][-1].item() != mmtokenizer.eoa:
            tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
            output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
    if i > 1:
        raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1] :]], dim=1)
    else:
        raw_output = output_seq

# endregion

# region Stage One - Postprocess

# save raw output and check sanity
ids = raw_output[0].cpu().numpy()
soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
if len(soa_idx) != len(eoa_idx):
    raise ValueError(f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}")

vocals = []
instrumentals = []
range_begin = 0
for i in range(range_begin, len(soa_idx)):
    codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
    if codec_ids[0] == 32016:
        codec_ids = codec_ids[1:]
    codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
    vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
    vocals.append(vocals_ids)
    instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
    instrumentals.append(instrumentals_ids)
vocals = np.concatenate(vocals, axis=1)
instrumentals = np.concatenate(instrumentals, axis=1)
vocal_save_path = os.path.join(
    stage1_output_dir,
    f"{random_id}:{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace(
        ".", "@"
    )
    + ".npy",
)
inst_save_path = os.path.join(
    stage1_output_dir,
    f"{random_id}:{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace(
        ".", "@"
    )
    + ".npy",
)
np.save(vocal_save_path, vocals)
np.save(inst_save_path, instrumentals)
stage1_output_set.append(vocal_save_path)
stage1_output_set.append(inst_save_path)

# endregion

# offload model

model.cpu()
del model
torch.cuda.empty_cache()

# region Stage Two - Load Model

print("Stage 2 inference...")
model_stage2 = AutoModelForCausalLM.from_pretrained(
    stage2_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="auto",
)
model_stage2.to(device)
model_stage2.eval()

if torch.__version__ >= "2.0.0":
    model_stage2 = torch.compile(model_stage2)

# endregion

# region Stage Two - Single generate


def stage2_generate(model, prompt, batch_size=16):
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
        codec_ids,
        global_offset=codectool.global_offset,
        codebook_size=codectool.codebook_size,
        num_codebooks=codectool.num_codebooks,
    ).astype(np.int32)

    # Prepare prompt_ids based on batch size or single input
    if batch_size > 1:
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * 300
            idx_end = (i + 1) * 300
            codec_list.append(codec_ids[:, idx_begin:idx_end])

        codec_ids = np.concatenate(codec_list, axis=0)
        prompt_ids = np.concatenate(
            [
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                codec_ids,
                np.tile([mmtokenizer.stage_2], (batch_size, 1)),
            ],
            axis=1,
        )
    else:
        prompt_ids = np.concatenate(
            [
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids.flatten(),  # Flatten the 2D array to 1D
                np.array([mmtokenizer.stage_2]),
            ]
        ).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]

    block_list = LogitsProcessorList(
        [BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)]
    )

    # Teacher forcing generate loop
    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx : frames_idx + 1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids

        with torch.no_grad():
            stage2_output = model.generate(
                input_ids=input_ids,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
            )

        assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, (
            f"output new tokens={stage2_output.shape[1] - prompt_ids.shape[1]}"
        )
        prompt_ids = stage2_output

    # Return output based on batch size
    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output


# endregion

# region Stage Two - Inference


def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))

        if os.path.exists(output_filename):
            print(f"{output_filename} stage2 has done.")
            continue

        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)

        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6

        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            output = stage2_generate(model, prompt[:, : output_duration * 50], batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)  # Adjust the last segment
                current_batch_size = (
                    batch_size if seg != num_segments - 1 or num_batch % batch_size == 0 else num_batch % batch_size
                )
                segment = stage2_generate(model, prompt[:, start_idx:end_idx], batch_size=current_batch_size)
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)

        # Process the ending part of the prompt
        if output_duration * 50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration * 50 :], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        output = codectool_stage2.ids2npy(output)

        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        # save output
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    return stage2_result


# endregion

stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=args.stage2_batch_size)
print(stage2_result)
print("Stage 2 DONE.\n")

# region Postprocess - Save Audio


# convert audio tokens to audio
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)


# endregion

# region Postprocess - Tracks Reconstruction

# reconstruct tracks
recons_output_dir = os.path.join(args.output_dir, "recons")
recons_mix_dir = os.path.join(recons_output_dir, "mix")
os.makedirs(recons_mix_dir, exist_ok=True)
tracks = []
for npy in stage2_result:
    codec_result = np.load(npy)
    decodec_rlt = []
    with torch.no_grad():
        decoded_waveform = codec_model.decode(
            torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device)
        )
    decoded_waveform = decoded_waveform.cpu().squeeze(0)
    decodec_rlt.append(torch.as_tensor(decoded_waveform))
    decodec_rlt = torch.cat(decodec_rlt, dim=-1)
    save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
    tracks.append(save_path)
    save_audio(decodec_rlt, save_path, 16000)

# endregion

# region Postprocess - Mixing

# mix tracks
for inst_path in tracks:
    try:
        print(inst_path)
        if (inst_path.endswith(".wav") or inst_path.endswith(".mp3")) and "_itrack" in inst_path:
            # find pair
            vocal_path = inst_path.replace("_itrack", "_vtrack")
            if not os.path.exists(vocal_path):
                continue
            # mix
            recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace("_itrack", "_mixed"))
            vocal_stem, sr = sf.read(inst_path)
            instrumental_stem, _ = sf.read(vocal_path)
            mix_stem = (vocal_stem + instrumental_stem) / 1
            sf.write(recons_mix, mix_stem, sr)
    except Exception as e:
        print(e)

# endregion

# region Postprocess - Upsample

# vocoder to upsample audios
vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
vocoder_output_dir = os.path.join(args.output_dir, "vocoder")
vocoder_stems_dir = os.path.join(vocoder_output_dir, "stems")
vocoder_mix_dir = os.path.join(vocoder_output_dir, "mix")
os.makedirs(vocoder_mix_dir, exist_ok=True)
os.makedirs(vocoder_stems_dir, exist_ok=True)
for npy in stage2_result:
    if "_itrack" in npy:
        # Process instrumental
        instrumental_output = process_audio(
            npy, os.path.join(vocoder_stems_dir, "itrack.mp3"), args.rescale, args, inst_decoder, codec_model
        )
    else:
        # Process vocal
        vocal_output = process_audio(
            npy, os.path.join(vocoder_stems_dir, "vtrack.mp3"), args.rescale, args, vocal_decoder, codec_model
        )

# endregion

# region Postprocess - Mixing

# mix tracks
try:
    mix_output = instrumental_output + vocal_output
    vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
    save_audio(mix_output, vocoder_mix, 44100, args.rescale)
    print(f"Created mix: {vocoder_mix}")
except RuntimeError as e:
    print(e)
    print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

# endregion

# region Postprocess - Last

# Post process
replace_low_freq_with_energy_matched(
    a_file=recons_mix,  # 16kHz
    b_file=vocoder_mix,  # 48kHz
    c_file=os.path.join(
        args.output_dir, os.path.basename(recons_mix) if not args.output_file else f"{args.output_file}.mp3"
    ),
    cutoff_freq=5500.0,
)

# endregion Postprocess - Last
