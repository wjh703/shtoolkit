import pathlib
from urllib.error import URLError
from urllib.request import urlretrieve

TN11E_URL = "https://filedrop.csr.utexas.edu/pub/slr/TN11E/TN11E.txt"
TN14_URL = (
    "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/gracefo/open/docs/TN-14_C30_C20_GSFC_SLR.txt"
)
TN13_CSR_URL = (
    "http://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/grace/open/docs/TN-13_GEOC_CSR_RL0602.txt"
)
TN13_JPL_URL = (
    "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/grace/open/docs/TN-13_GEOC_JPL_RL0601.txt"
)
TN13_GFZ_URL = (
    "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/grace/open/docs/TN-13_GEOC_GFZ_RL0601.txt"
)
ICE6G_D_URL = r"https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2F2016JB013844&file=jgrb52450-sup-0003-Data_S3.txt"


DOWNLOAD_LIST = [TN11E_URL, TN14_URL, TN13_CSR_URL, TN13_JPL_URL, TN13_GFZ_URL]


def download_technical_notes():
    for url in DOWNLOAD_LIST:
        filename = pathlib.Path(url).name
        local_filepath = pathlib.Path(__file__).absolute().parent / "data" / filename
        print(f"Downloading {filename}")
        try:
            urlretrieve(url, local_filepath)
            print(f"Finish {filename}")
        except URLError:
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            urlretrieve(url, local_filepath)


if __name__ == "__main__":
    download_technical_notes()
