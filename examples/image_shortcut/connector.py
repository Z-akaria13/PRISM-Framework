"""
PRISM connector for a folder of images and a CSV, the common case for imaging models.
Copy it into your project and change the marked lines.

    images/<id>.png   one file per row of the CSV, named by its ID column
    model.pt          the model's weights (torch.save(model.state_dict(), ...))
"""
import os

import numpy as np
import torch
from PIL import Image

from prism import PrismConnector

from model import ScanNet  # CHANGE: your model class, from your own code in the project folder


class ImageFolderConnector(PrismConnector):
    # The labels in your CSV's target column, in the order of your model's output columns.
    class_names = ["healthy", "disease"]  # CHANGE

    def load_model(self):
        model = ScanNet()  # CHANGE
        weights = torch.load(os.path.join(self.project_root, "model.pt"), map_location="cpu")  # CHANGE: file name
        model.load_state_dict(weights)
        return model.eval()

    def load_item(self, row_id, metadata_row):
        # row_id is the row's value in the ID column chosen in the Evidence tab, as text (e.g. "0042").
        path = os.path.join(self.project_root, "images", f"{row_id}.png")  # CHANGE: folder and extension
        pixels = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255  # CHANGE: your preprocessing
        return torch.from_numpy(pixels).unsqueeze(0)  # 1 × H × W

    def run_inference(self, model, batch_data):
        return model(torch.stack(batch_data))  # one row of logits per item

    def render_item_for_gallery(self, raw_data_item):
        return {"type": "image", "payload": raw_data_item[0].numpy()}
