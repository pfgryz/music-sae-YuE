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
        --max_new_tokens 1000 \
        --repetition_penalty 1.1

generate-few:
    #!/bin/bash
    cd inference

    paths=(
        g0
        g1
        g2
        g3
        g4
        g5
    )

    for path in "${paths[@]}"
    do
        uv run infer.py \
            --cuda_idx 0 \
            --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
            --stage2_model m-a-p/YuE-s2-1B-general \
            --genre_txt ../examples/$path.txt \
            --lyrics_txt ../examples/lyrics.txt \
            --run_n_segments 2 \
            --stage2_batch_size 4 \
            --output_dir output \
            --max_new_tokens 3000 \
            --repetition_penalty 1.1
    done