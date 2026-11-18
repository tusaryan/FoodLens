from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

LIGHT_FOLDERS = [
    "RED",
    "GREEN",
    "BLUE"
]


# ============================================================
# SELECT exp1_Sorted FOLDER
# ============================================================

def select_folder():

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_folder = filedialog.askdirectory(
        title="Select exp1_Sorted Folder"
    )

    root.destroy()

    return selected_folder


# ============================================================
# ANALYSE ONE IMAGE
# ============================================================

def analyse_image(image_path):

    try:

        # Open image ONLY for reading
        # Original image is NOT changed
        with Image.open(image_path) as image:

            # Convert only in memory for RGB analysis
            image_rgb = image.convert("RGB")

            image_array = np.array(
                image_rgb,
                dtype=np.float64
            )

        # Separate RGB channels
        red_channel = image_array[:, :, 0]
        green_channel = image_array[:, :, 1]
        blue_channel = image_array[:, :, 2]

        # Calculate mean values
        mean_r = np.mean(red_channel)
        mean_g = np.mean(green_channel)
        mean_b = np.mean(blue_channel)

        return (
            round(mean_r, 7),
            round(mean_g, 7),
            round(mean_b, 7)
        )

    except Exception as error:

        print(
            f"ERROR ANALYSING: {image_path.name}"
        )

        print(error)

        return None


# ============================================================
# GET IMAGES FROM FOLDER
# ============================================================

def get_images(folder):

    images = []

    if not folder.exists():
        return images

    # Search only images
    for file in folder.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() in IMAGE_EXTENSIONS:

            images.append(file)

    # Sort images according to filename
    return sorted(
        images,
        key=lambda x: x.name
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 60)
    print("IMAGE RGB MEAN INTENSITY ANALYSIS")
    print("=" * 60)

    # Select exp1_Sorted folder
    selected_folder = select_folder()

    if not selected_folder:

        print("No folder selected.")
        return

    main_folder = Path(selected_folder)

    print()
    print("SELECTED FOLDER:")
    print(main_folder)

    print()
    print("=" * 60)
    print("ANALYSING RED / GREEN / BLUE FOLDERS")
    print("=" * 60)

    # Results for each light
    results_by_light = {
        "RED": [],
        "GREEN": [],
        "BLUE": []
    }

    total_images = 0


    # ========================================================
    # PROCESS RED / GREEN / BLUE
    # ========================================================

    for light in LIGHT_FOLDERS:

        light_folder = main_folder / light

        print()
        print("-" * 60)
        print(f"PROCESSING: {light}")
        print(light_folder)
        print("-" * 60)

        # Get all images
        images = get_images(
            light_folder
        )

        print(
            f"IMAGES FOUND: {len(images)}"
        )


        # ----------------------------------------------------
        # ANALYSE EACH IMAGE
        # ----------------------------------------------------

        for sequence, image_path in enumerate(
            images,
            start=1
        ):

            means = analyse_image(
                image_path
            )

            if means is None:
                continue

            mean_r, mean_g, mean_b = means

            # Only required columns
            result = {

                "No.": sequence,

                "Image Name": image_path.name,

                "Mean_R": mean_r,

                "Mean_G": mean_g,

                "Mean_B": mean_b
            }

            results_by_light[
                light
            ].append(
                result
            )

            total_images += 1

            print(
                f"{light} | "
                f"No. {sequence} | "
                f"{image_path.name}"
            )


    # ========================================================
    # CHECK IF IMAGES FOUND
    # ========================================================

    if total_images == 0:

        print()
        print("=" * 60)
        print("NO IMAGES FOUND")
        print("=" * 60)

        print()
        print(
            "Select the exp1_Sorted folder containing:"
        )

        print("RED")
        print("GREEN")
        print("BLUE")

        return


    # ========================================================
    # CREATE EXCEL FILE
    # ========================================================

    excel_path = (
        main_folder /
        "Image_Intensity_Results.xlsx"
    )


    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:


        # ====================================================
        # CREATE RED / GREEN / BLUE SHEETS
        # ====================================================

        for light in LIGHT_FOLDERS:

            # Create dataframe
            light_df = pd.DataFrame(
                results_by_light[light],
                columns=[
                    "No.",
                    "Image Name",
                    "Mean_R",
                    "Mean_G",
                    "Mean_B"
                ]
            )


            # Save to Excel sheet
            light_df.to_excel(
                writer,
                sheet_name=light,
                index=False
            )


        # ====================================================
        # FORMAT EXCEL SHEETS
        # ====================================================

        workbook = writer.book


        for sheet_name in LIGHT_FOLDERS:

            worksheet = workbook[
                sheet_name
            ]


            # Freeze first row
            worksheet.freeze_panes = "A2"


            # Add filter
            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )


            # Column widths

            # No.
            worksheet.column_dimensions[
                "A"
            ].width = 10


            # Image Name
            worksheet.column_dimensions[
                "B"
            ].width = 60


            # Mean_R
            worksheet.column_dimensions[
                "C"
            ].width = 15


            # Mean_G
            worksheet.column_dimensions[
                "D"
            ].width = 15


            # Mean_B
            worksheet.column_dimensions[
                "E"
            ].width = 15


    # ========================================================
    # COMPLETED
    # ========================================================

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        f"TOTAL IMAGES ANALYSED: {total_images}"
    )

    print()

    for light in LIGHT_FOLDERS:

        print(
            f"{light} IMAGES: "
            f"{len(results_by_light[light])}"
        )


    print()

    print("EXCEL FILE CREATED:")

    print(excel_path)

    print()

    print(
        "Excel columns:"
    )

    print(
        "No. | Image Name | Mean_R | Mean_G | Mean_B"
    )

    print()

    print(
        "Original images were NOT modified."
    )

    print(
        "No crop, resize or re-saving was performed."
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()