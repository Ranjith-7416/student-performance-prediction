
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
from io import BytesIO

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

print("Connecting to UCI Machine Learning Repository...")

try:
    request = Request(
        URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urlopen(request, timeout=30) as response:
        outer_data = response.read()

    print("Download successful!")

    # Open the outer ZIP
    with ZipFile(BytesIO(outer_data)) as outer_zip:

        print("Outer ZIP files:", outer_zip.namelist())

        # Find the nested ZIP
        nested_name = next(
            (name for name in outer_zip.namelist()
             if Path(name).name == "student.zip"),
            None
        )

        if nested_name:
            nested_data = outer_zip.read(nested_name)

            # Open the inner ZIP
            with ZipFile(BytesIO(nested_data)) as inner_zip:
                csv_name = next(
                    (name for name in inner_zip.namelist()
                     if Path(name).name == "student-mat.csv"),
                    None
                )

                if not csv_name:
                    raise FileNotFoundError(
                        "student-mat.csv missing from inner ZIP"
                    )

                output = DATA_DIR / "student-mat.csv"
                output.write_bytes(inner_zip.read(csv_name))

        else:
            # Handle an archive containing CSV directly
            csv_name = next(
                (name for name in outer_zip.namelist()
                 if Path(name).name == "student-mat.csv"),
                None
            )

            if not csv_name:
                raise FileNotFoundError(
                    "student-mat.csv not found in downloaded archive"
                )

            output = DATA_DIR / "student-mat.csv"
            output.write_bytes(outer_zip.read(csv_name))

    print("\nSUCCESS! Dataset downloaded.")
    print("Saved location:", output)
    print("File size:", output.stat().st_size, "bytes")

except Exception as error:
    print("\nERROR:", repr(error))
    print("Please share this complete error message.")
