"""Build six title-free PPT thumbnails and an ordered 3 x 2 panel."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STILL_ROOT = PROJECT_ROOT / "Test" / "paper_stills" / "training"
SELECTED = STILL_ROOT / "selected_layout_b"
OUTPUT = STILL_ROOT / "scenario_overview_s1_s6"
FULLRES = OUTPUT / "source_full_resolution"

THUMBNAIL_SIZE = (960, 540)
GAP_PX = 18

SOURCES = {
    "S1": (
        SELECTED / "S1_WZ1_layoutB_training.png",
        SELECTED / "S1_WZ1_layoutB_training_manifest.json",
    ),
    "S2": (
        STILL_ROOT
        / "s2_wz1_b__20260904_150550_664"
        / "training_s2_wz1_b_origin0_overview.png",
        STILL_ROOT / "s2_wz1_b__20260904_150550_664" / "paper_still_manifest.json",
    ),
    "S3": (
        SELECTED / "S3_WZ1_layoutB_training.png",
        SELECTED / "S3_WZ1_layoutB_training_manifest.json",
    ),
    "S4": (
        SELECTED / "S4_WZ1_layoutB_training.png",
        SELECTED / "S4_WZ1_layoutB_training_manifest.json",
    ),
    "S5": (
        SELECTED / "S5_WZ1_layoutB_training.png",
        SELECTED / "S5_WZ1_layoutB_training_manifest.json",
    ),
    "S6": (
        STILL_ROOT
        / "s6_wz1_b__20260904_150730_850"
        / "training_s6_wz1_b_origin0_overview.png",
        STILL_ROOT / "s6_wz1_b__20260904_150730_850" / "paper_still_manifest.json",
    ),
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FULLRES.mkdir(parents=True, exist_ok=True)

    thumbnails: dict[str, Image.Image] = {}
    for scenario, (source, manifest) in SOURCES.items():
        if not source.exists() or not manifest.exists():
            raise FileNotFoundError(f"Missing source pair for {scenario}: {source}")
        with Image.open(source) as image:
            image = image.convert("RGB")
            if image.size != (1920, 1080):
                raise ValueError(f"Unexpected source dimensions for {scenario}: {image.size}")
            thumb = image.resize(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            thumb.save(OUTPUT / f"{scenario}.png", format="PNG", optimize=True)
            thumbnails[scenario] = thumb
        shutil.copy2(source, FULLRES / f"{scenario}_fullres.png")
        shutil.copy2(manifest, FULLRES / f"{scenario}_manifest.json")

    panel_width = 3 * THUMBNAIL_SIZE[0] + 2 * GAP_PX
    panel_height = 2 * THUMBNAIL_SIZE[1] + GAP_PX
    panel = Image.new("RGB", (panel_width, panel_height), (255, 255, 255))
    for index, scenario in enumerate(("S1", "S2", "S3", "S4", "S5", "S6")):
        row, column = divmod(index, 3)
        x = column * (THUMBNAIL_SIZE[0] + GAP_PX)
        y = row * (THUMBNAIL_SIZE[1] + GAP_PX)
        panel.paste(thumbnails[scenario], (x, y))

    panel.save(OUTPUT / "S1_to_S6_3x2.png", format="PNG", optimize=True)
    panel.save(
        OUTPUT / "S1_to_S6_3x2_preview.jpg",
        format="JPEG",
        quality=90,
        subsampling=0,
        optimize=True,
    )
    print(f"Wrote six {THUMBNAIL_SIZE[0]}x{THUMBNAIL_SIZE[1]} title-free thumbnails")
    print(f"Wrote ordered panel: {panel_width}x{panel_height}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
