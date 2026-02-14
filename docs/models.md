More information on the expected JSON schema for the configuration can be found in [splifft.config.Config][].

## Supported models

### BS Roformer

- checkpoint: [`roformer-fp16.pt`](https://github.com/undef13/splifft/releases/download/v0.0.1/roformer-fp16.pt)
- configuration: [`bs_roformer.json`](https://github.com/undef13/splifft/blob/main/data/config/bs_roformer.json)
- [`config.model`][splifft.models.bs_roformer.BSRoformerParams]
- [`config.model_type = "bs_roformer"`][splifft.config.Config.model_type]

## Visualizations

The following are quick comparisons for the quality of different models, as evaluated on MVSep. Support for running these models are not yet implemented but may come in a future release.

<!--
https://github.com/undef13/splifft/blob/main/scripts/mvsep.py
uv run scripts/mvsep.py --instruments instrum --id 7534 --id 7573 --id 7768 --id 8257 --id 8303 --id 8393 --id 8362
uv run scripts/mvsep.py --instruments vocals --id 7475 --id 7706 --id 8093 --id 8265 --id 8337 --id 8377
-->

=== "Instrumental"

    ![Instrumental](./assets/mvsep/plots/correlations_instrum.svg)

=== "Vocals"

    ![Vocals](./assets/mvsep/plots/correlations_vocals.svg)

### Beat This

This is an audio-to-signal model. The output is a continuous activation curves where peaks represent beat candidates (logits) at 50 Hz.

- [`config.model`][splifft.models.beat_this.BeatThisParams]
- [`config.model_type = "beat_this"`][splifft.config.Config.model_type]

The reference package `beat_this` depends on `madmom.features.downbeats.DBNDownBeatTrackingProcessor` as a preprocessing step, which uses:

- a state space to represent progression through a measure. for a 4/4 time signature, states represent grid positions (e.g. "beat 1, 25% through")
- transition model encodes tempo (bpm) and continuity, penalising sudden tempo jumps
- observation model takes the raw logit values (activations) as input probabilties
- using a Viterbi-like algorithm to find the most likely path through states

However [`madmom` has been abandoned as of 2024-08-25](https://github.com/CPJKU/madmom/issues/553) and so we do not depend on it.
