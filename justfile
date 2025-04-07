default: prepare-env

prepare-env:
    uvx pre-commit install
    uv sync

generate-example:
    #!/bin/sh
    cd inference
    uv run infer.py \
        --cuda_idx 0 \
        --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
        --stage2_model m-a-p/YuE-s2-1B-general \
        --genre_txt ../examples/genre.txt \
        --lyrics_txt ../examples/lyrics.txt \
        --run_n_segments 2 \
        --stage2_batch_size 4 \
        --output_dir output \
        --max_new_tokens 300 \
        --repetition_penalty 1.1