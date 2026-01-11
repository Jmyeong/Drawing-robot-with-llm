import json
import os
import random
from glob import glob
from tqdm import tqdm

DATA_ROOT = "/ssd1/jm_data/quickdraw"
SAVE_DIR = "dataset/quickdraw_multi_stroke"  
os.makedirs(SAVE_DIR, exist_ok=True)

PROMPT_TEMPLATES = [
    "Draw a '{category}' and provide the point coordinates.",
    "Generate the coordinate list for drawing a '{category}'.",
    "Create a vector sketch of a '{category}'.",
    "Give me the stroke points to draw a '{category}'.",
    "How do I draw a '{category}'? Show me the coordinates.",
    "What are the vector points for a '{category}'?",
    "Can you predict the drawing path for a '{category}'?",
    "I need a coordinate sequence to render a '{category}'.",
    "Predict the stroke points required to visualize a '{category}'.",
    "Output the 2D coordinates representing a '{category}'.",
    "Provide the path data for a '{category}' sketch.",
    "Coordinates for '{category}'.",
    "Draw '{category}'.",
    "Sketch '{category}' in vector points."
]

def downsample_strokes(strokes, step=2):
    new_strokes = []
    for stroke in strokes:
        sampled = stroke[::step]
        if len(sampled) > 0:
            new_strokes.append(sampled)
    return new_strokes

def process_multi_strokes(drawing):
    strokes_list = []
    for xs, ys in drawing:
        single_stroke = []
        for x, y in zip(xs, ys):
            single_stroke.append([float(x), float(y)])
        strokes_list.append(single_stroke)
    return strokes_list


def process_ndjson_file(ndjson_path):
    filename = os.path.basename(ndjson_path)
    category = os.path.splitext(filename)[0]

    save_path = os.path.join(SAVE_DIR, f"{category}.jsonl")

    with open(save_path, "w", encoding="utf-8") as fout:
        with open(ndjson_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                strokes = process_multi_strokes(item["drawing"])
                
                strokes = downsample_strokes(strokes, step=1)

                coord_json = json.dumps({"strokes": strokes}, ensure_ascii=False)
                
                selected_template = random.choice(PROMPT_TEMPLATES)
                prompt = selected_template.format(category=category)

                ex = {
                    "messages": [
                        {"role": "user",
                         "content": [{"type": "text", "text": prompt}]},
                        {"role": "assistant",
                         "content": [{"type": "text", "text": coord_json}]}
                    ]
                }

                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")


def convert_all():
    ndjson_files = sorted(glob(os.path.join(DATA_ROOT, "**/*.ndjson"), recursive=True))
    print(f"[INFO] Found {len(ndjson_files)} ndjson files")

    for path in tqdm(ndjson_files):
        process_ndjson_file(path)

    print("🔥 ALL DONE! Saved to:", SAVE_DIR)


if __name__ == "__main__":
    convert_all()