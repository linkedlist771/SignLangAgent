from pathlib import Path


ROOT = Path(__file__).parent.parent

CHECKPOINTS_DIR = ROOT / "checkpoints"

CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    from loguru import logger

    logger.debug(f"ROOT: \n{ROOT}")
