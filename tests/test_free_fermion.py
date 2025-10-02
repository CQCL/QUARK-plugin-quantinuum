from pathlib import Path
import quark
from .conftest import TEST_DIR


def test_free_fermion_config():
    config_path = TEST_DIR / Path("configs/free_fermion_test_config.yml")
    quark.start(["-c", str(config_path)])
