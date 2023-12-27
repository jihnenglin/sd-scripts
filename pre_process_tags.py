import os
import html

root_dir = "~/sd-train"
scraped_data_dir = os.path.join(root_dir, "train_data")

def pre_process_tags(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.endswith(".txt"):
            old_path = item_path
            new_file_name = os.path.splitext(os.path.splitext(item)[0])[0] + ".txt"
            new_path = os.path.join(directory, new_file_name)

            os.rename(old_path, new_path)

            with open(new_path, "r") as f:
                contents = f.read()

            contents = html.unescape(contents)
            contents = contents.replace("_", " ")
            lines = contents.split("\n")
            if len(lines) > 1:
                contents = ", ".join(lines[:-1])

                with open(new_path, "w") as f:
                    f.write(contents)

        elif os.path.isdir(item_path):
            pre_process_tags(item_path)

pre_process_tags(scraped_data_dir)
