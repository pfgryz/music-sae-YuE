import os
import os.path
import shutil
import re

print(os.getcwd())
directory = "./inference/ablate"
pattern = re.compile(r"^(.+?)-(.+?)-(\[.+\])(\.\w+)$")

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        continue

    segments = filename.replace(".mp3", "").split("-")
    _, category, *other = segments

    output_directory_name = "pure" if len(other) == 0 else f"layer/{other[0]}"
    output_directory = os.path.join(directory, output_directory_name)
    output_filepath = os.path.join(output_directory, filename)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    shutil.move(filepath, output_filepath)
    print(f"MOVED {filename} | {filepath} -> {output_filepath}")
