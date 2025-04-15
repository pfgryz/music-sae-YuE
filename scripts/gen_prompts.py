import os

output_directory = "./examples/genres"
os.makedirs(output_directory, exist_ok=True)

with open("./examples/prompts.txt", "r") as handle:
    lines = handle.readlines()

for i, line in enumerate(lines):
    with open(os.path.join(output_directory, f"g{i}.txt"), "w") as handle:
        handle.write(line)
