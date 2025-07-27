import os
import yaml

mlruns_dir = "mlruns"

for root, dirs, files in os.walk(mlruns_dir):
    if "meta.yaml" in files:
        path = os.path.join(root, "meta.yaml")
        with open(path, "r") as f:
            meta = yaml.safe_load(f)

        if meta.get("source", "").startswith("file:///"):
            run_id = meta.get("run_id")
            exp_id = path.split(os.sep)[1]  # assumes mlruns/<exp_id>/...
            rel_path = f"file:./mlruns/{exp_id}/{run_id}/artifacts/model"
            meta["source"] = rel_path
            meta["storage_location"] = rel_path

            with open(path, "w") as f:
                yaml.safe_dump(meta, f)

            print(f"Fixed: {path}")
