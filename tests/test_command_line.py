import sys
from unittest.mock import patch

import pytest

from src.models.command_line_args import CommandLineArgs
from src.runtime.command_line import CommandLine


def test_parse_arguments_ingest_monkey(monkeypatch):
    test_args = ["script_name", "ingest", "--config", "config/ingest_config.yaml"]
    monkeypatch.setattr(sys, "argv", test_args)

    args = CommandLine.parse_arguments()
    assert args.command == "ingest"
    assert args.config == "config/ingest_config.yaml"
    assert args.debug is False


def test_parse_arguments_train_monkey(monkeypatch):
    test_args = ["script_name", "train", "--debug"]
    monkeypatch.setattr(sys, "argv", test_args)

    args = CommandLine.parse_arguments()
    assert args.command == "train"
    assert args.debug is True  # Fixed: --debug should be True


def test_parse_arguments_ingest_mock():
    test_args = ["script_name", "ingest", "--config", "config/ingest_config.yaml"]
    with patch("sys.argv", test_args):
        args = CommandLine.parse_arguments()

    assert args.command == "ingest"
    assert args.config == "config/ingest_config.yaml"
    assert args.debug is False


def test_parse_arguments_train_mock():
    test_args = ["script_name", "train", "--debug"]
    with patch("sys.argv", test_args):
        args = CommandLine.parse_arguments()

    assert args.command == "train"
    assert args.debug is True


def test_parse_arguments_ingest_no_config(monkeypatch):
    test_args = ["script_name", "ingest"]
    monkeypatch.setattr(sys, "argv", test_args)

    args = CommandLine.parse_arguments()
    assert args.command == "ingest"
    assert args.config is None  # Default when not provided
    assert args.debug is False


def test_parse_arguments_train_no_debug(monkeypatch):
    test_args = ["script_name", "train"]
    monkeypatch.setattr(sys, "argv", test_args)

    args = CommandLine.parse_arguments()
    assert args.command == "train"
    assert args.config == "config/model_config.yaml"  # Ensure default config is used
    assert args.debug is False


def test_parse_arguments_ingest_with_debug(monkeypatch):
    test_args = ["script_name", "ingest", "--debug"]
    monkeypatch.setattr(sys, "argv", test_args)

    args = CommandLine.parse_arguments()
    assert args.command == "ingest"
    assert args.config is None
    assert args.debug is True


def test_parse_arguments_no_command(monkeypatch):
    test_args = ["script_name"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit):  # Should exit with error
        CommandLine.parse_arguments()
