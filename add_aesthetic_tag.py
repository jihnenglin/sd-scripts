import json
from tqdm import tqdm

with open(f"aesthetic_score.json", "r") as f:
    aesthetic_scores = json.load(f)

aesthetic_thresholds = [6.0, 5.66, 5.33, 4.0]
aesthetic_tag_names = ["best aesthetic", "amazing aesthetic", "great aesthetic", "normal aesthetic", "bad aesthetic"]
assert len(aesthetic_tag_names) == len(aesthetic_thresholds) + 1, \
       "The number of `aesthetic_tag_names` should be equal to the number of `aesthetic_thresholds` plus one"

def get_tag_name(score, thresholds, tag_names):
    not_last = False
    for j in range(len(thresholds)):
        if score >= thresholds[j]:
            not_last = True
            break
    if not_last:
        tag = tag_names[j]
    else:
        tag = tag_names[-1]
    return tag

for img_path, score in tqdm(aesthetic_scores.items(), smoothing=0.0):
    tag_path = img_path.split(".")[0] + ".txt"
    
    aesthetic_tag = get_tag_name(aesthetic_scores[img_path], aesthetic_thresholds, aesthetic_tag_names)
    #print(aesthetic_scores[img_path], aesthetic_tag)
    #input("press enter to continue")
    with open(tag_path, "r") as f:
        original = f.read()
    with open(tag_path, "w") as f:
        f.write(f"{aesthetic_tag} ||| {original}")
