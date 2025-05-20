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
        --stage2_batch_size 128 \
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
            --stage2_batch_size 128 \
            --output_dir output \
            --output_file $path-equal \
            --max_new_tokens 1000 \
            --repetition_penalty 1.1
    done

ablate-example:
    #!/bin/sh
    cd inference
    uv run infer_ablate.py \
        --cuda_idx 0 \
        --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
        --stage2_model m-a-p/YuE-s2-1B-general \
        --genre_txt ../examples/genre.txt \
        --lyrics_txt ../examples/lyrics.txt \
        --run_n_segments 2 \
        --stage2_batch_size 128 \
        --output_dir output \
        --output_file ablate-eg \
        --max_new_tokens 1000 \
        --repetition_penalty 1.1 \
        --ablate \
        --ablation-layer 1

ablate-ref:
    #!/bin/bash
    cd inference

    for path in {15..105}
    do
        uv run infer.py \
            --cuda_idx 1 \
            --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
            --stage2_model m-a-p/YuE-s2-1B-general \
            --genre_txt ../examples/genres/g$path.txt \
            --lyrics_txt ../examples/lyrics.txt \
            --run_n_segments 2 \
            --stage2_batch_size 128 \
            --output_dir ablate \
            --output_file $path-ref \
            --max_new_tokens 300 \
            --repetition_penalty 1.1
    done

ablate-few:
    #!/bin/bash
    cd inference

    for path in {64..105}
    do
        for layer in {0..31}
        do
            uv run infer_ablate.py \
                --cuda_idx 0 \
                --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
                --stage2_model m-a-p/YuE-s2-1B-general \
                --genre_txt ../examples/genres/g$path.txt \
                --lyrics_txt ../examples/lyrics.txt \
                --run_n_segments 2 \
                --stage2_batch_size 128 \
                --output_dir ablate \
                --output_file $path-ablate-$layer \
                --max_new_tokens 300 \
                --repetition_penalty 1.1 \
                --ablate \
                --ablation-layer $layer
        done
    done

ablation-fad generations_dir score_path:
    #!/bin/sh
    cd dependencies/fadtk
    for item in $(seq 0 31); do 
        uv run fadtk --inf clap-laion-audio fma_pop {{ generations_dir }}/layer/$item {{ score_path }};
    done
    uv run fadtk --inf clap-laion-audio fma_pop {{ generations_dir }}/pure {{ score_path }};

ablation-relative-fad generations_dir score_path:
    #!/bin/sh
    cd dependencies/fadtk
    for item in $(seq 0 31); do 
        uv run fadtk --inf clap-laion-audio {{ generations_dir }}/pure {{ generations_dir }}/layer/$item {{ score_path }};
    done

test-activation:
    #!/bin/sh
    cd inference
    uv run infer_stage1.py

trace-test:
    #!/bin/bash
    cd inference

    uv run infer_trace.py \
                --cuda_idx 1 \
                --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
                --stage2_model m-a-p/YuE-s2-1B-general \
                --genre_txt ../examples/genres/g0.txt \
                --lyrics_txt ../examples/lyrics.txt \
                --run_n_segments 2 \
                --stage2_batch_size 128 \
                --output_dir other \
                --output_file trace \
                --max_new_tokens 300 \
                --repetition_penalty 1.1 \
                --ablate \
                --ablation-layer 13