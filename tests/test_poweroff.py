from agentenv_zeroad.train import poweroff


def test_power_off_finds_shutdown_in_sbin_even_on_the_box_scripts_path(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")
    looked: list[str | None] = []
    monkeypatch.setattr(poweroff.shutil, "which", lambda name, path=None: looked.append(path) or f"/usr/sbin/{name}")
    calls: list[list[str]] = []
    assert poweroff.power_off(lambda argv: calls.append(argv) or 0)
    assert calls == [["/usr/sbin/shutdown", "-h", "now"]] and "/usr/sbin" in (looked[0] or "")


def test_power_off_falls_back_to_systemctl_and_reports_when_nothing_works(monkeypatch):
    monkeypatch.setattr(poweroff.shutil, "which", lambda name, path=None: f"/usr/bin/{name}")
    calls: list[list[str]] = []
    assert poweroff.power_off(lambda argv: calls.append(argv) or (1 if argv[0].endswith("shutdown") else 0))
    assert calls[-1] == ["/usr/bin/systemctl", "poweroff"]
    assert not poweroff.power_off(lambda argv: 1)
