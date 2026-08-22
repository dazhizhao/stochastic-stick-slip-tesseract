from types import SimpleNamespace

import pytest

from scripts import reproduce


def test_reproduce_runs_smoke_before_renderer(monkeypatch, capsys):
    commands = []

    def successful_run(command, **kwargs):
        commands.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(reproduce.subprocess, "run", successful_run)
    reproduce.main()

    assert [command for command, _ in commands] == [
        reproduce.SMOKE_COMMAND,
        reproduce.RENDER_COMMAND,
    ]
    assert all(kwargs["cwd"] == reproduce.ROOT for _, kwargs in commands)
    output = capsys.readouterr().out
    assert "Direct AD physics gradient: zero" in output
    assert "CRN-FD physics gradient: nonzero" in output
    assert "One optimizer update: completed" in output
    assert "outputs/jumpgrad_visuals/" in output


def test_reproduce_propagates_child_failure(monkeypatch, capsys):
    commands = []

    def failing_run(command, **kwargs):
        commands.append(tuple(command))
        return SimpleNamespace(returncode=7, stdout="partial output\n", stderr="failure\n")

    monkeypatch.setattr(reproduce.subprocess, "run", failing_run)
    with pytest.raises(SystemExit, match="7"):
        reproduce.main()

    assert commands == [reproduce.SMOKE_COMMAND]
    error = capsys.readouterr().err
    assert "JumpGrad smoke audit failed" in error
    assert "partial output" in error
    assert "failure" in error
