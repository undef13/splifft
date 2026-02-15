# ruff: noqa: E402
PATH_MIXTURE = "data/audio/input/3BFTio5296w.flac"

from splifft.inference import InferenceEngine

engine = InferenceEngine.from_pretrained(
    config_path="data/config/bs_roformer.json",
    checkpoint_path="data/models/roformer-fp16.pt",
)
result = engine.run(PATH_MIXTURE)
print(result)

#
# alternatively, to track progress for long files or on slow hardware:
#

import logging

from splifft.inference import InferenceOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)

for event in engine.stream(PATH_MIXTURE):
    if isinstance(event, InferenceOutput):
        print(event)
        break
    logger.info(event)
