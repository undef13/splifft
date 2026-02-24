# Models

`splifft` supports:

- Separation: isolate stems (`vocals`, `drums`, `bass`, ...)
- Sequence labeling: predict frame-wise musical signals (`beat`, `pitch`, `onset`, ...)

Goal: one runtime, one config style, one cache/registry flow. More information on config shape is in [`splifft.config.Config`][].

## Supported models

### BS-Roformer / Mel-Roformer

- [`config.model`][splifft.models.bs_roformer.BSRoformerParams]
- [`config.model_type = "bs_roformer" or "mel_roformer"`][splifft.config.Config.model_type]

Use this family when separation quality is the top priority.

In `splifft`, Mel-Roformer is represented through the same model family with [`splifft.models.bs_roformer.MelBandsConfig`][] enabled.

### MDX23C

- [`config.model`][splifft.models.mdx23c.MDX23CParams]
- [`config.model_type = "mdx23c"`][splifft.config.Config.model_type]

This is an older architecture that won the [Sound Demixing Challenge 2023 Leaderboard C](https://transactions.ismir.net/articles/10.5334/tismir.171). It is used in some drum separation checkpoints from the community registry.

### Beat This

- [`config.model`][splifft.models.beat_this.BeatThisParams]
- [`config.model_type = "beat_this"`][splifft.config.Config.model_type]

Outputs frame-wise beat/downbeat activations (`.npy`). We intentionally avoid depending on legacy DBN post-processing stacks so inference stays lightweight and reproducible.

??? "Why not add DBN?"

    The reference package `beat_this` depends on `madmom.features.downbeats.DBNDownBeatTrackingProcessor` as a preprocessing step, which uses:

    - a state space to represent progression through a measure. for a 4/4 time signature, states represent grid positions (e.g. "beat 1, 25% through")
    - transition model encodes tempo (bpm) and continuity, penalising sudden tempo jumps
    - observation model takes the raw logit values (activations) as input probabilties
    - using a Viterbi-like algorithm to find the most likely path through states

    However [`madmom` has been abandoned as of 2024-08-25](https://github.com/CPJKU/madmom/issues/553) and so we do not depend on it.

### PESTO

- [`config.model`][splifft.models.pesto.PestoParams]
- [`config.model_type = "pesto"`][splifft.config.Config.model_type]

Use PESTO when the input is dominated by a single melodic source (voice, lead, solo line). It is designed for stable frame-level F0 tracking.

Outputs `pitch`, `confidence`, `volume`, `activations` as frame sequences (`.npy`).

### Basic Pitch (polyphonic pitch)

- [`config.model`][splifft.models.basic_pitch.BasicPitchParams]
- [`config.model_type = "basic_pitch"`][splifft.config.Config.model_type]

Use Basic Pitch when multiple notes can be active at once. In `splifft` we intentionally expose raw outputs and do not support MIDI decoding, so downstream applications can choose their own thresholding/hysteresis policy.

Outputs `onset`, `note`, and `contour` activation maps (`.npy`). Except `contour` (3 bins per semitone), all others have 1 bin per semitone.

## Visual comparisons (MVSep)

The following are quick comparisons for model quality on MVSep (separation-only).

<!--
https://github.com/undef13/splifft/blob/main/scripts/mvsep.py
uv run scripts/mvsep.py correlations --instruments instrum --id 7534 --id 7573 --id 7768 --id 8257 --id 8303 --id 8393 --id 8362 --id 9505 --id 9475 --id 9580
uv run scripts/mvsep.py correlations --instruments vocals --id 7475 --id 7706 --id 8093 --id 8265 --id 8337 --id 8377 --id 9470
-->

=== "Instrumental"

    ![Instrumental](./assets/mvsep/plots/correlations_instrum.svg)

=== "Vocals"

    ![Vocals](./assets/mvsep/plots/correlations_vocals.svg)
