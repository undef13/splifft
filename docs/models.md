More information on the expected JSON schema for the configuration can be found in [`splifft.config.Config`][].

## Supported models

To view the list of models in the default registry:

```command
$ splifft ls
id                                          size     created_at  purpose        outputs                  
mel_roformer-gabox-flowers_v10              489.6M   2026-01-04  separation     instrum                  
mel_roformer-becruily-deux                  435.0M   2025-12-29  separation     vocals,instrum           
mel_roformer-gabox-inst_fv9                 913.1M   2025-12-23  separation     instrum                  
bs_roformer-unwa-hyperace_v2_vocals         288.7M   2025-12-20  separation     vocals                   
bs_roformer-unwa-hyperace_v2_instrum        288.7M   2025-12-18  separation     instrum                  
mel_roformer-gabox-inst_fv7b                913.0M   2025-11-22  separation     instrum                  
mel_roformer-gabox-voc_fv7_beta3            913.0M   2025-11-22  separation     vocals                   
mel_roformer-gabox-voc_fv7_beta2            913.0M   2025-11-13  separation     vocals
...
```

### BS Roformer

This is an audio-to-audio spectrogram-masking model.

- Checkpoint: [`roformer-fp16.pt`](https://github.com/undef13/splifft/releases/download/v0.0.1/roformer-fp16.pt)
- Configuration: [`bs_roformer.json`](https://github.com/undef13/splifft/blob/main/data/config/bs_roformer.json)
- [`config.model`][splifft.models.bs_roformer.BSRoformerParams]
- [`config.model_type = "bs_roformer" or "mel_roformer"`][splifft.config.Config.model_type]

In `splifft`, Mel Roformers are BS Roformers with the [`splifft.models.bs_roformer.MelBandsConfig`][] set.

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
