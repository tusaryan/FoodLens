from pathlib import Path
from datetime import datetime
import re
import csv
import shutil


# ============================================================
# SETTINGS
# ============================================================

# ENTER ONLY THE MAIN "expriment" FOLDER PATH HERE
MAIN_FOLDER = r"C:\Users\rohit\Downloads\expriment\EXP-4(05.09.2026)"


# ============================================================
# LIGHT SETTINGS
# ============================================================

# Each light stays ON for 1 minute = 60 seconds
LIGHT_DURATION = 60

# Light sequence
LIGHTS = [
    "RED",
    "GREEN",
    "BLUE"
]


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# READ DATE AND TIME FROM IMAGE FILENAME
# ============================================================

def get_photo_time(image_path):

    """
    Example:

    IMG_20260831_133919.jpg

    Date: 2026-08-31
    Time: 13:39:19
    """

    filename = image_path.stem

    # Find:
    # 8-digit date + "_" + 6-digit time
    match = re.search(
        r"(\d{8})_(\d{6})",
        filename
    )

    if not match:
        return None

    date_part = match.group(1)
    time_part = match.group(2)

    try:

        return datetime.strptime(
            date_part + time_part,
            "%Y%m%d%H%M%S"
        )

    except ValueError:

        return None


# ============================================================
# FIND ACTIVE LIGHT
# ============================================================

def get_light(elapsed_seconds):

    """
    0:00 to 0:59     RED
    1:00 to 1:59     GREEN
    2:00 to 2:59     BLUE
    3:00 to 3:59     RED

    Then repeat.
    """

    light_index = (
        int(elapsed_seconds // LIGHT_DURATION)
        % len(LIGHTS)
    )

    return LIGHTS[light_index]


# ============================================================
# FORMAT EXPERIMENT TIME
# ============================================================

def format_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    secs = seconds % 60

    return (
        f"{hours:02}:"
        f"{minutes:02}:"
        f"{secs:02}"
    )


# ============================================================
# PROCESS ONE EXPERIMENT IMAGE FOLDER
# ============================================================

def process_image_folder(exp_folder):

    print()
    print("=" * 70)
    print("PROCESSING:")
    print(exp_folder)
    print("=" * 70)


    # --------------------------------------------------------
    # CREATE SORTED FOLDER BESIDE ORIGINAL FOLDER
    # --------------------------------------------------------

    sorted_folder = (
        exp_folder.parent
        / f"{exp_folder.name}_Sorted"
    )

    sorted_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # Create RED / GREEN / BLUE folders
    for light in LIGHTS:

        (
            sorted_folder
            / light
        ).mkdir(
            exist_ok=True
        )


    # --------------------------------------------------------
    # FIND VALID IMAGES DIRECTLY INSIDE EXPERIMENT FOLDER
    # --------------------------------------------------------

    photos = []

    for file in exp_folder.iterdir():

        if not file.is_file():
            continue

        if (
            file.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue


        # Read original date/time from filename
        photo_time = get_photo_time(file)

        if photo_time is None:

            print(
                f"SKIPPED (time not found in filename): "
                f"{file.name}"
            )

            continue


        photos.append(
            (file, photo_time)
        )


    # --------------------------------------------------------
    # SORT FROM EARLIEST TO LATEST
    # --------------------------------------------------------

    photos.sort(
        key=lambda item: item[1]
    )


    print()
    print(
        f"VALID PHOTOS FOUND: "
        f"{len(photos)}"
    )


    if len(photos) == 0:

        print("No valid photos found.")

        return


    # --------------------------------------------------------
    # EARLIEST PHOTO = EXPERIMENT START
    # --------------------------------------------------------

    experiment_start = photos[0][1]

    print()
    print(
        "EXPERIMENT START:"
    )

    print(
        experiment_start.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print(
        "STARTING LIGHT: RED"
    )


    # --------------------------------------------------------
    # CREATE CSV LOG
    # --------------------------------------------------------

    csv_path = (
        sorted_folder
        / "experiment_log.csv"
    )


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )


        writer.writerow([
            "Sequence",
            "Original Filename",
            "Original Date",
            "Original Time",
            "Experiment Start",
            "Experiment Time",
            "Elapsed Seconds",
            "Light",
            "Sorted Filename"
        ])


        # ----------------------------------------------------
        # PROCESS EACH PHOTO
        # ----------------------------------------------------

        for sequence, (
            photo,
            photo_time
        ) in enumerate(
            photos,
            start=1
        ):


            # Calculate time since experiment started
            elapsed_seconds = (
                photo_time
                - experiment_start
            ).total_seconds()


            # Experiment elapsed time
            experiment_time = (
                format_time(
                    elapsed_seconds
                )
            )


            # Find active light
            light = get_light(
                elapsed_seconds
            )


            # ------------------------------------------------
            # NEW FILE NAME
            #
            # Example:
            #
            # 0001_RED_2026-08-31_13-39-19_EXP-00-00-00.jpg
            # ------------------------------------------------

            new_filename = (
                f"{sequence:04}_"
                f"{light}_"
                f"{photo_time.strftime('%Y-%m-%d')}_"
                f"{photo_time.strftime('%H-%M-%S')}_"
                f"EXP-{experiment_time.replace(':', '-')}"
                f"{photo.suffix.lower()}"
            )


            # Output location
            output_photo = (
                sorted_folder
                / light
                / new_filename
            )


            # ------------------------------------------------
            # COPY ORIGINAL IMAGE WITHOUT MODIFYING IT
            #
            # shutil.copy2() does NOT:
            # - resize image
            # - recompress image
            # - change pixels
            # - change image quality
            # ------------------------------------------------

            try:

                shutil.copy2(
                    photo,
                    output_photo
                )


                # Write information to CSV
                writer.writerow([
                    sequence,

                    photo.name,

                    photo_time.strftime(
                        "%d-%m-%Y"
                    ),

                    photo_time.strftime(
                        "%H:%M:%S"
                    ),

                    experiment_start.strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),

                    experiment_time,

                    int(
                        elapsed_seconds
                    ),

                    light,

                    new_filename
                ])


                print(
                    f"{sequence:04} | "
                    f"{photo_time.strftime('%H:%M:%S')} | "
                    f"{experiment_time} | "
                    f"{light} | "
                    f"{photo.name}"
                )


            except Exception as error:

                print(
                    f"ERROR COPYING: "
                    f"{photo.name}"
                )

                print(error)


    print()
    print("COMPLETED:")
    print(sorted_folder)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    main_folder = Path(
        MAIN_FOLDER
    )


    # Check main folder
    if not main_folder.exists():

        print()
        print("=" * 70)
        print("ERROR: MAIN FOLDER NOT FOUND")
        print(MAIN_FOLDER)
        print("=" * 70)

        return


    print()
    print("=" * 70)
    print("SEARCHING FOR EXPERIMENT IMAGE FOLDERS")
    print("=" * 70)


    processed_folders = 0


    # --------------------------------------------------------
    # SEARCH ALL FOLDERS
    # --------------------------------------------------------

    for folder in main_folder.rglob("*"):

        if not folder.is_dir():
            continue


        # ----------------------------------------------------
        # IGNORE SORTED FOLDERS
        # AND EVERYTHING INSIDE THEM
        # ----------------------------------------------------

        if "_SORTED" in folder.name.upper():
            continue


        if any(
            "_SORTED" in parent.name.upper()
            for parent in folder.parents
        ):
            continue


        # ----------------------------------------------------
        # CHECK WHETHER THIS FOLDER DIRECTLY CONTAINS IMAGES
        # ----------------------------------------------------

        contains_images = any(

            file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS

            for file in folder.iterdir()

        )


        # ----------------------------------------------------
        # PROCESS FOLDER
        # ----------------------------------------------------

        if contains_images:

            processed_folders += 1

            process_image_folder(
                folder
            )


    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL PROCESSING COMPLETED")
    print(
        f"TOTAL IMAGE FOLDERS PROCESSED: "
        f"{processed_folders}"
    )
    print("=" * 70)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()