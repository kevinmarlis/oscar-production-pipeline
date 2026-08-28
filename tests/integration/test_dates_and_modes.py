"""Orchestration glue: date selection, output-mode resolution, path construction.

These are the pieces a structural refactor is most likely to churn, and none of them are a
black box -- they are date arithmetic, file-existence logic, and string manipulation, so the
assertions verify *correctness* from first principles rather than echoing current output.
"""

import os

import numpy as np

from oscar.utils import data_utils as du


def _touch(path, nbytes=100):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * nbytes)


# --- extract_date / get_file_path ------------------------------------

def test_extract_date_from_filename():
    assert du.extract_date("/data/oscar_currents_final_19930101.nc") == ("1993", "01")


def test_get_file_path_final_and_interim(monkeypatch, tmp_path):
    monkeypatch.setattr(du, "OUTPUT_DIR", str(tmp_path / "OUT"))
    final = du.get_file_path("1993-01-05", "final")
    interim = du.get_file_path("1993-01-05", "interim")
    assert final.endswith("/FINAL/1993/01/oscar_currents_final_19930105.nc")
    assert interim.endswith("/INTERIM/1993/01/oscar_currents_interim_19930105.nc")


# --- get_existing_dates / get_dates_to_process -----------------------

def test_get_existing_dates_scans_year_month_tree(tmp_path):
    root = tmp_path / "OUT"
    _touch(str(root / "1993" / "01" / "oscar_currents_final_19930103.nc"))
    _touch(str(root / "1993" / "01" / "oscar_currents_final_19930107.nc"))

    existing = du.get_existing_dates(str(root), "1993-01-01", "1993-01-10")
    assert sorted(str(d) for d in existing) == ["1993-01-03", "1993-01-07"]


def test_get_dates_to_process_skips_existing_final(tmp_path):
    # get_dates_to_process scans <OUTPUT_DIR>/FINAL for already-computed days.
    out = tmp_path / "OUT"
    _touch(str(out / "FINAL" / "1993" / "01" / "oscar_currents_final_19930102.nc"))

    all_dates, to_process = du.get_dates_to_process(
        np.datetime64("1993-01-01"), np.datetime64("1993-01-04"), str(out), override=False
    )
    assert all_dates == ["1993-01-01", "1993-01-02", "1993-01-03", "1993-01-04"]
    # 01-02 already has a final file -> excluded.
    assert to_process == ["1993-01-01", "1993-01-03", "1993-01-04"]


def test_get_dates_to_process_override_returns_all(tmp_path):
    out = tmp_path / "OUT"
    _touch(str(out / "FINAL" / "1993" / "01" / "oscar_currents_final_19930102.nc"))

    all_dates, to_process = du.get_dates_to_process(
        np.datetime64("1993-01-01"), np.datetime64("1993-01-04"), str(out), override=True
    )
    # override recomputes everything regardless of existing files.
    assert to_process == all_dates
    assert to_process == ["1993-01-01", "1993-01-02", "1993-01-03", "1993-01-04"]


# --- determine_oscar_mode --------------------------------------------

def test_determine_oscar_mode_final_interim_none(tmp_path, monkeypatch):
    sst = tmp_path / "sst"
    wind = tmp_path / "wind"
    ssh_final = tmp_path / "ssh_final"
    ssh_interim = tmp_path / "ssh_interim"
    for d in (sst, wind, ssh_final, ssh_interim):
        d.mkdir()
    monkeypatch.setattr(du, "SST_SRC_DIR", str(sst))
    monkeypatch.setattr(du, "WIND_SRC_DIR", str(wind))
    monkeypatch.setattr(du, "SSH_SRC_FINAL_DIR", str(ssh_final))
    monkeypatch.setattr(du, "SSH_SRC_INTERIM_DIR", str(ssh_interim))

    # day 1: sst + wind + ssh-final present -> "final"
    _touch(str(sst / "20200101_x.nc"))
    _touch(str(wind / "20200101_x.nc"))
    _touch(str(ssh_final / "20200101_x.nc"))
    # day 2: sst + wind + ssh-interim (no final) -> "interim"
    _touch(str(sst / "20200102_x.nc"))
    _touch(str(wind / "20200102_x.nc"))
    _touch(str(ssh_interim / "20200102_x.nc"))
    # day 3: missing SST -> "None"
    _touch(str(wind / "20200103_x.nc"))
    _touch(str(ssh_final / "20200103_x.nc"))

    modes = du.determine_oscar_mode(
        ["2020-01-01", "2020-01-02", "2020-01-03"], "cmems"
    )
    assert modes == ["final", "interim", "None"]


def test_determine_oscar_mode_missing_ssh_is_none(tmp_path, monkeypatch):
    sst = tmp_path / "sst"
    wind = tmp_path / "wind"
    ssh_final = tmp_path / "ssh_final"
    ssh_interim = tmp_path / "ssh_interim"
    for d in (sst, wind, ssh_final, ssh_interim):
        d.mkdir()
    monkeypatch.setattr(du, "SST_SRC_DIR", str(sst))
    monkeypatch.setattr(du, "WIND_SRC_DIR", str(wind))
    monkeypatch.setattr(du, "SSH_SRC_FINAL_DIR", str(ssh_final))
    monkeypatch.setattr(du, "SSH_SRC_INTERIM_DIR", str(ssh_interim))

    # SST + WIND present but no SSH anywhere -> None.
    _touch(str(sst / "20200101_x.nc"))
    _touch(str(wind / "20200101_x.nc"))
    assert du.determine_oscar_mode(["2020-01-01"], "cmems") == ["None"]


def test_determine_oscar_mode_rejects_bad_ssh_mode(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="ssh_mode must be"):
        du.determine_oscar_mode(["2020-01-01"], "bogus")
