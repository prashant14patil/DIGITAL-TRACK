import json

from digitaltrack.cli import main


def test_cli_synth_validate_reconstruct_roundtrip(tmp_path, capsys):
    out = tmp_path / "s"
    assert main(["synth", "--shape", "cylinder", "--out", str(out), "--views", "12"]) == 0
    assert main(["validate", str(out)]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["reconstruct", str(out)]) == 0
    runs = list((out / "processing").iterdir())
    assert len(runs) == 1 and json.loads((runs[0] / "run_log.json").read_text())["stage"] == "tsdf_reconstruction"


def test_cli_validate_reports_and_fails_on_a_broken_session(tmp_path, capsys):
    out = tmp_path / "s"
    main(["synth", "--out", str(out), "--views", "4"])
    capsys.readouterr()
    import os, stat
    p = out / "frames/frame_000001/depth_raw.bin"
    os.chmod(p, stat.S_IWRITE)
    p.unlink()
    assert main(["validate", str(out)]) == 1
    assert "missing_file" in capsys.readouterr().out


def test_cli_reconstruct_refuses_an_invalid_session(tmp_path, capsys):
    assert main(["reconstruct", str(tmp_path / "nothing")]) == 1
    assert "failed validation" in capsys.readouterr().out
