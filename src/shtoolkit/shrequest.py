import pathlib
from urllib.error import URLError
from urllib.request import urlretrieve

TN11E_URL = "https://filedrop.csr.utexas.edu/pub/slr/TN11E/TN11E.txt"
TN14_URL = (
    "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/grace/open/docs/TN-14_C30_C20_GSFC_SLR.txt"
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


def download_technical_notes():
    for k, v in globals().items():
        if "TN" not in k or "URL" not in k:
            continue
        url = v
        filename = url.split("/")[-1]
        local_filepath = pathlib.Path(__file__).absolute().parent / "data" / filename
        print(f"Downloading {filename}")
        try:
            urlretrieve(url, local_filepath)
        except URLError:
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            urlretrieve(url, local_filepath)
