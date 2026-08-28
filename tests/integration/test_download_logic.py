"""Download skip/overwrite branches and output-path construction, with all network mocked.

We patch the network entry points that ``oscar.utils.download`` binds at import time
(``subset`` from copernicusmarine, ``cdsapi.Client``, ``subprocess.run``, ``requests.get``)
and redirect the directory constants to a tmp path. No credentials or network are used, and
there is no live-download test.
"""

import os

import pytest

from oscar.utils import download

MIN_BYTES = download.MIN_BYTES
DATE = "1993-01-01"
D8 = "19930101"


def _write(path, nbytes=MIN_BYTES):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"0" * nbytes)


# --- CMEMS SSH --------------------------------------------------------

@pytest.fixture
def cmems_dirs(tmp_path, monkeypatch):
    final = tmp_path / "final"
    interim = tmp_path / "interim"
    monkeypatch.setattr(download, "SSH_SRC_FINAL_DIR", str(final))
    monkeypatch.setattr(download, "SSH_SRC_INTERIM_DIR", str(interim))
    return final, interim


def test_cmems_skips_when_final_exists_and_no_overwrite(cmems_dirs, monkeypatch):
    final, _ = cmems_dirs
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", False)
    _write(str(final / "1993" / "01" / f"ssh_{D8}.nc"))

    called = []
    monkeypatch.setattr(download, "subset", lambda **kw: called.append(kw))

    download.download_ssh_cmems([DATE])
    assert called == []  # existing final + no overwrite -> no download attempted


def test_cmems_downloads_final_when_missing(cmems_dirs, monkeypatch):
    final, _ = cmems_dirs
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", False)

    calls = []

    def fake_subset(**kw):
        calls.append(kw)
        _write(os.path.join(kw["output_directory"], kw["output_filename"]))

    monkeypatch.setattr(download, "subset", fake_subset)
    download.download_ssh_cmems([DATE])

    assert len(calls) == 1
    kw = calls[0]
    assert kw["dataset_id"] == download.FINAL_ID
    assert kw["output_filename"] == f"ssh_{D8}.nc"
    assert kw["output_directory"] == str(final / "1993" / "01")
    assert os.path.exists(final / "1993" / "01" / f"ssh_{D8}.nc")


def test_cmems_overwrite_redownloads_existing_final(cmems_dirs, monkeypatch):
    final, _ = cmems_dirs
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", True)
    _write(str(final / "1993" / "01" / f"ssh_{D8}.nc"))

    calls = []

    def fake_subset(**kw):
        calls.append(kw)
        _write(os.path.join(kw["output_directory"], kw["output_filename"]))

    monkeypatch.setattr(download, "subset", fake_subset)
    download.download_ssh_cmems([DATE])
    assert len(calls) == 1 and calls[0]["dataset_id"] == download.FINAL_ID


def test_cmems_falls_back_to_interim_when_final_empty(cmems_dirs, monkeypatch):
    final, interim = cmems_dirs
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", False)

    calls = []

    def fake_subset(**kw):
        calls.append(kw["dataset_id"])
        # FINAL returns an undersized (invalid) file; INTERIM returns a valid one.
        nbytes = 100 if kw["dataset_id"] == download.FINAL_ID else MIN_BYTES
        _write(os.path.join(kw["output_directory"], kw["output_filename"]), nbytes)

    monkeypatch.setattr(download, "subset", fake_subset)
    download.download_ssh_cmems([DATE])

    assert calls == [download.FINAL_ID, download.INTERIM_ID]
    assert os.path.exists(interim / "1993" / "01" / f"ssh_{D8}.nc")


# --- ERA5 WIND --------------------------------------------------------

def test_era5_skip_and_path(tmp_path, monkeypatch):
    wind_dir = tmp_path / "wind"
    monkeypatch.setattr(download, "WIND_SRC_DIR", str(wind_dir))
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", False)

    retrieve_calls = []

    class FakeClient:
        def retrieve(self, name, request, target):
            retrieve_calls.append((name, request, target))
            _write(target)

    monkeypatch.setattr(download.cdsapi, "Client", lambda *a, **k: FakeClient())

    # First run: file missing -> downloads to era5_YYYYMMDD.nc under year/month.
    download.download_wind_era5([DATE])
    out_file = wind_dir / "1993" / "01" / f"era5_{D8}.nc"
    assert len(retrieve_calls) == 1
    name, request, target = retrieve_calls[0]
    assert name == "reanalysis-era5-single-levels"
    assert target == str(out_file)
    assert request["year"] == "1993" and request["month"] == "01" and request["day"] == "01"
    assert os.path.exists(out_file)

    # Second run: file now exists, no overwrite -> no further retrieve.
    download.download_wind_era5([DATE])
    assert len(retrieve_calls) == 1


# --- CMC SST (subprocess CLI) ----------------------------------------

def test_sst_collection_and_path_pre_2016(tmp_path, monkeypatch):
    sst_dir = tmp_path / "sst"
    monkeypatch.setattr(download, "SST_SRC_DIR", str(sst_dir))
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", False)

    run_calls = []

    def fake_run(cmd, **kw):
        run_calls.append(cmd)
        target_folder = cmd[cmd.index("-d") + 1]
        _write(os.path.join(target_folder, f"{D8}120000-CMC-L4.nc"))

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    download.download_sst_cmc([DATE])

    assert len(run_calls) == 1
    cmd = run_calls[0]
    # 1993 is before 2016 -> old CMC v2.0 collection.
    assert "CMC0.2deg-CMC-L4-GLOB-v2.0" in cmd
    assert str(sst_dir / "1993" / "01") in cmd


def test_sst_skips_when_present_and_no_overwrite(tmp_path, monkeypatch):
    sst_dir = tmp_path / "sst"
    monkeypatch.setattr(download, "SST_SRC_DIR", str(sst_dir))
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", False)
    _write(str(sst_dir / "1993" / "01" / f"{D8}120000-CMC-L4.nc"))

    run_calls = []
    monkeypatch.setattr(download.subprocess, "run", lambda cmd, **kw: run_calls.append(cmd))

    download.download_sst_cmc([DATE])
    assert run_calls == []


# --- DRIFTER (requests) ----------------------------------------------

def test_drifter_skipped_when_validation_off(tmp_path, monkeypatch):
    drifter_dir = tmp_path / "drifters"
    monkeypatch.setattr(download, "DRIFTER_SRC_DIR", str(drifter_dir))
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", False)
    monkeypatch.setattr(download, "DO_VALIDATION", False)

    get_calls = []
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: get_calls.append(a))

    download.get_drifter_data("1993-01")
    assert get_calls == []  # no validation -> no network call


def test_drifter_downloads_when_validation_on(tmp_path, monkeypatch):
    drifter_dir = tmp_path / "drifters"
    monkeypatch.setattr(download, "DRIFTER_SRC_DIR", str(drifter_dir))
    monkeypatch.setattr(download, "OVERWRITE_DOWNLOAD", False)
    monkeypatch.setattr(download, "DO_VALIDATION", True)

    class FakeResponse:
        content = b"0" * MIN_BYTES

        def raise_for_status(self):
            pass

    get_urls = []

    def fake_get(url, *a, **k):
        get_urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(download.requests, "get", fake_get)
    path = download.get_drifter_data("1993-01")

    assert len(get_urls) == 1
    assert "drifter_6hour_qc" in get_urls[0]
    expected = drifter_dir / "drifter_6hour_qc_199301.nc"
    assert path == str(expected)
    assert os.path.exists(expected)
