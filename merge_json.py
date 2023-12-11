import json

# Make json file from parallel latent caching

NUM_OF_DEVICES = 4

json_path = "~/sd-train/json/meta_lat.json"

merged_metadata = {}
for i in range(NUM_OF_DEVICES):
    with open(f"{json_path[:-5]}_{i}.json", "r") as f:
        merged_metadata.update(json.load(f))

print(f"metadata: {len(merged_metadata)} entries")
with open(json_path, "w") as f:
    json.dump(merged_metadata, f, indent=2)
