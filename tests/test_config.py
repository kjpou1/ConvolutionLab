from src.config.config import Config


# This is a hack to make sure the Config object is reset
def reset_config():
    Config.reset()


def test_config_initialization():
    assert not Config.is_initialized()  # Ensure not initialized initially
    Config.initialize()
    assert Config.is_initialized()  # Ensure it is initialized after calling initialize
    reset_config()


def test_config_create_initialization():
    assert not Config.is_initialized()  # Ensure not initialized initially
    Config()
    assert Config.is_initialized()  # Ensure it is initialized after calling initialize
    reset_config()
