"""Build a labeled visual index for the selected Layout-B paper stills.

The source screenshots are opened read-only and are never modified.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent / "paper_stills" / "training" / "selected_layout_b"
OUTPUT = ROOT / "contact_sheet_layoutB.png"

SCENARIOS = ("S1", "S3", "S4", "S5")
WORKZONES = ("WZ1", "WZ2", "WZ3")

CANVAS_WIDTH = 1920
MARGIN_X = 28
MARGIN_BOTTOM = 24
ROW_LABEL_WIDTH = 88
GAP_X = 12
GAP_Y = 12
TITLE_HEIGHT = 66
COLUMN_HEADER_HEIGHT = 45
CELL_LABEL_HEIGHT = 31

COLORS = {
    "background": "#0B1017",
    "panel": "#141B24",
    "panel_alt": "#101720",
    "border": "#334155",
    "text": "#F8FAFC",
    "muted": "#A9B6C6",
    "accent": "#19C3FF",
    "na": "#1A222D",
    "na_line": "#283442",
}


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf") if bold else Path(r"C:\Windows\Fonts\segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(26, bold=True)
SUBTITLE_FONT = load_font(16)
COLUMN_FONT = load_font(21, bold=True)
ROW_FONT = load_font(20, bold=True)
CELL_FONT = load_font(17, bold=True)
NA_FONT = load_font(20, bold=True)
NA_SMALL_FONT = load_font(15)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=fill)


def main() -> None:
    cell_width = (
        CANVAS_WIDTH
        - 2 * MARGIN_X
        - ROW_LABEL_WIDTH
        - (len(SCENARIOS) - 1) * GAP_X
    ) // len(SCENARIOS)
    image_height = round(cell_width * 9 / 16)
    row_height = image_height + CELL_LABEL_HEIGHT
    canvas_height = (
        TITLE_HEIGHT
        + COLUMN_HEADER_HEIGHT
        + len(WORKZONES) * row_height
        + (len(WORKZONES) - 1) * GAP_Y
        + MARGIN_BOTTOM
    )

    canvas = Image.new("RGB", (CANVAS_WIDTH, canvas_height), COLORS["background"])
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (MARGIN_X, 11),
        "Training Scenario Screenshot Index — Spectator Layout B",
        font=TITLE_FONT,
        fill=COLORS["text"],
    )
    draw.text(
        (MARGIN_X, 43),
        "Illustrative stills for paper-figure selection; not evaluation rollouts",
        font=SUBTITLE_FONT,
        fill=COLORS["muted"],
    )

    grid_x = MARGIN_X + ROW_LABEL_WIDTH
    header_y = TITLE_HEIGHT
    for col, scenario in enumerate(SCENARIOS):
        x = grid_x + col * (cell_width + GAP_X)
        centered_text(
            draw,
            (x, header_y, x + cell_width, header_y + COLUMN_HEADER_HEIGHT),
            f"Scenario {scenario}",
            COLUMN_FONT,
            COLORS["accent"],
        )

    grid_y = TITLE_HEIGHT + COLUMN_HEADER_HEIGHT
    for row, workzone in enumerate(WORKZONES):
        y = grid_y + row * (row_height + GAP_Y)
        centered_text(
            draw,
            (MARGIN_X, y, MARGIN_X + ROW_LABEL_WIDTH - 10, y + row_height),
            workzone,
            ROW_FONT,
            COLORS["text"],
        )

        for col, scenario in enumerate(SCENARIOS):
            x = grid_x + col * (cell_width + GAP_X)
            image_box = (x, y, x + cell_width, y + image_height)
            label_box = (
                x,
                y + image_height,
                x + cell_width,
                y + row_height,
            )

            draw.rectangle(image_box, fill=COLORS["panel"], outline=COLORS["border"], width=2)
            draw.rectangle(label_box, fill=COLORS["panel_alt"], outline=COLORS["border"], width=1)

            if scenario == "S4" and workzone == "WZ3":
                # Draw the hatch on a cell-sized image so diagonal strokes are
                # clipped at the S4/WZ3 boundary and cannot spill into S3/WZ3.
                na_panel = Image.new("RGB", (cell_width, image_height), COLORS["na"])
                na_draw = ImageDraw.Draw(na_panel)
                for offset in range(-image_height, cell_width, 26):
                    na_draw.line(
                        (offset, image_height, offset + image_height, 0),
                        fill=COLORS["na_line"],
                        width=2,
                    )
                canvas.paste(na_panel, (x, y))
                draw.rectangle(image_box, outline=COLORS["border"], width=2)
                centered_text(draw, image_box, "N/A", NA_FONT, COLORS["muted"])
                centered_text(
                    draw,
                    (x, y + image_height // 2 + 20, x + cell_width, y + image_height),
                    "Scenario S4 has no WZ3",
                    NA_SMALL_FONT,
                    COLORS["muted"],
                )
            else:
                source = ROOT / f"{scenario}_{workzone}_layoutB_training.png"
                if not source.exists():
                    raise FileNotFoundError(f"Missing selected screenshot: {source}")
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    image.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
                    paste_x = x + (cell_width - image.width) // 2
                    paste_y = y + (image_height - image.height) // 2
                    canvas.paste(image, (paste_x, paste_y))
                draw.rectangle(image_box, outline=COLORS["border"], width=2)

            centered_text(
                draw,
                label_box,
                f"{scenario} / {workzone}",
                CELL_FONT,
                COLORS["text"],
            )

    canvas.save(OUTPUT, format="PNG", optimize=True)
    print(f"Wrote: {OUTPUT}")
    print(f"Dimensions: {canvas.width} x {canvas.height}")


if __name__ == "__main__":
    main()
