import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

# Indices into the 26-dim state/action vectors:
# 7-12: right joint positions, 6: right gripper
# 20-25: left joint positions, 19: left gripper
XARM_SLICE_INDICES = [7, 8, 9, 10, 11, 12, 6, 20, 21, 22, 23, 24, 25, 19]


def make_xarm_example() -> dict:
    """Creates a random input example for the XArm policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class XArmInputs(transforms.DataTransformFn):
    """Inputs for the dual XArm policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [26] (will be sliced to 14) or [14] (already sliced)
    - actions: [action_horizon, 26] or [action_horizon, 14]
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        # Process state: slice from 26-dim to 14-dim if needed.
        state = np.asarray(data["state"])
        if state.shape[-1] == 26:
            state = state[..., XARM_SLICE_INDICES]

        # Process images.
        in_images = data["images"]

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        images_dict = {name: convert_image(img) for name, img in in_images.items()}

        # Map to model image slots.
        base_image = images_dict["cam_high"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in images_dict:
                images[dest] = images_dict[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] == 26:
                actions = actions[..., XARM_SLICE_INDICES]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XArmOutputs(transforms.DataTransformFn):
    """Outputs for the dual XArm policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}
