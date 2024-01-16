import json
import dateutil.parser
import glob
from tqdm import tqdm

folder = "~/sd-train/train_data"

def generate_tags(data):
    def process_tags(tag_str):
        processed_tags = []
        for tag_name in tag_str.split(" "):
            if len(tag_name) > 3:
                tag_name = tag_name.replace("_", " ")
            processed_tags.append(tag_name)
        return ", ".join(processed_tags)

    created_at = data.get("media_asset", {}).get("created_at", "")
    try:
        parsed_date = dateutil.parser.isoparse(created_at)
        year = parsed_date.year
        year_tag = f"year {year}"
    except (ValueError, AttributeError):
        print("Invalid or missing created_at date.")
        year_tag = ""

    rating = data.get("rating")
    score = data.get("score")

    tags_general = process_tags(data.get("tag_string_general", ""))
    tags_character = process_tags(data.get("tag_string_character", ""))
    tags_copyright = process_tags(data.get("tag_string_copyright", ""))
    tags_artist =  process_tags(data.get("tag_string_artist", ""))
    tags_meta =  process_tags(data.get("tag_string_meta", ""))

    quality_tag = ""
    if score >= 150:
        quality_tag = "best quality"
    elif 100 <= score < 150:
        quality_tag = "amazing quality"
    elif 75 <= score < 100:
        quality_tag = "great quality"
    elif 0 <= score < 75 :
        quality_tag = "normal quality"
    elif -5 < score < 0:
        quality_tag = "bad quality"
    elif score <= -5:
        quality_tag = "worst quality"

    if rating in "q":
        nsfw_tags = "fairly nsfw"
    elif rating in "e":
        nsfw_tags = "very nsfw"
    elif rating in "s":
        nsfw_tags = "slightly nsfw"
    else:
        nsfw_tags = ""

    tags_general_list = tags_general.split(', ')
    special_tags = [
        "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple boys", "male focus",
        "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple girls",
    ]

    found_special_tags = [tag for tag in tags_general_list if tag in special_tags]

    for tag in found_special_tags:
        tags_general_list.remove(tag)

    first_general_tag = ', '.join(found_special_tags)
    rest_general_tags = ', '.join(tags_general_list)

    tags_separator = "|||"

    pre_separator_tags = []
    post_separator_tags = []

    if first_general_tag:
        pre_separator_tags.append(first_general_tag)
    if tags_character:
        pre_separator_tags.append(tags_character)
    if tags_copyright:
        pre_separator_tags.append(tags_copyright)
    if tags_artist:
        pre_separator_tags.append(tags_artist)

    if rest_general_tags:
        post_separator_tags.append(rest_general_tags)
    if tags_meta:
        post_separator_tags.append(tags_meta)
    if nsfw_tags:
        post_separator_tags.append(nsfw_tags)
    if year_tag:
        post_separator_tags.append(year_tag)
    if quality_tag:
        post_separator_tags.append(quality_tag)

    pre_separator_str = ', '.join(pre_separator_tags)
    post_separator_str = ', '.join(post_separator_tags)

    caption = f"{pre_separator_str}, {tags_separator} {post_separator_str}"

    #print(caption)
    #print()
    return caption

if __name__ == "__main__":
    json_files = glob.glob(f"{folder}/*.json")
    for file_path in tqdm(json_files, smoothing=0.0):
        with open(file_path, "r") as f:
            data = json.load(f)
        tags = generate_tags(data)

        tag_file_path = file_path.split(".")[0] + ".txt"
        #print(tag_file_path)
        with open(tag_file_path, "w") as f:
            f.write(tags)
        #input("Press enter to continue")
