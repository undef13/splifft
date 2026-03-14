# ruff: noqa: E402
PATH_MIXTURE = "data/audio/input/3BFTio5296w.flac"

from splifft.inference import InferenceEngine

engine = InferenceEngine.from_pretrained(
    config="/path/to/config.json",
    checkpoint_path="/path/to/checkpoint.pt",
)
result = engine.run(PATH_MIXTURE)
print(result)

#
# or, if you use the default user cache registry
#

engine = InferenceEngine.from_registry("bs_roformer-fruit-sw")

#
# and to track progress for long files or on slow hardware:
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
