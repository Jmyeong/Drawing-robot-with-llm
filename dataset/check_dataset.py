import json
import matplotlib.pyplot as plt

def draw_points_from_jsonl(jsonl_path, index=0):
    """
    jsonl_path: JSONL 파일 경로
    index: 시각화할 샘플 번호
    """

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if index >= len(lines):
        raise ValueError("index가 파일 길이를 초과했습니다.")

    data = json.loads(lines[index])

    coord_json = data["messages"][-1]["content"][0]["text"]
    coord_dict = json.loads(coord_json)

    points = coord_dict["points"]   # [[x, y], [x, y], ...]

    if len(points) == 0:
        raise ValueError("points가 비어 있습니다!")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(6, 6))

    plt.plot(xs, ys, marker='o', linewidth=2, markersize=3)

    plt.gca().invert_yaxis()  
    plt.axis("equal")
    plt.grid(True)
    plt.title(f"Polyline Visualization #{index}")
    plt.show()


if __name__ == "__main__":
    draw_points_from_jsonl(
        "dataset/quickdraw/The Effiel Tower.jsonl",
        index=4
    )
