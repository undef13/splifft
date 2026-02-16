More information on the expected JSON schema for the configuration can be found in [`splifft.config.Config`][].

## Supported models

To view the list of all models in the default registry (including those that are not locally available):

```command
$ splifft ls -a
model_id                                           size  created_at  purpose        target_instruments     
mel_roformer-unwa-big_beta7                      944.7M  2026-02-13  separation     vocals                 
bs_roformer-anvuew-dereverb                      204.5M  2026-02-13  dereverb       dry                    
mel_roformer-jasper-vocalsandeffects                  ?  2026-02-01  separation     vocals                 
mdx23c-jasper-dnr                                     ?  2026-02-01  separation     foreground,background  
mel_roformer-gabox-voc_fv7                       489.6M  2026-01-27  separation     vocals                 
mel_roformer-unwa-large_inst_v2                  238.2M  2026-01-25  separation     instrum           
mel_roformer-chencfd-guitar                      276.7M  2026-01-23  separation     guitar                 
scnet-aname-4stems_huge_v1                    ⤓  261.4M  2026-01-11  separation     drums,bass,other,vocals
mel_roformer-gabox-flowers_v10                   489.6M  2026-01-04  separation     instrum           
...
```

To pull the model's weights and configuration to the default user cache directory (optionally pass `-f` to force overwrite):

See the [platformdirs](https://platformdirs.readthedocs.io/en/latest/platforms.html#user-cache-dir) documentation for the default location. On Linux / MacOS, this will default to your `XDG_CACHE_HOME`.

```command
$ splifft pull bs_roformer-fruit-sw
[00:00:42] INFO     wrote config for 'bs_roformer-fruit-sw' at         io.py:186
                    /home/undef13/.cache/splifft/bs_roformer-fruit-sw
                    /config.json                                                      
           INFO     pulling weights for 'bs_roformer-fruit-sw'         io.py:195
[00:00:48] INFO     verified digest                                    io.py:276
                    'sha256:06fd1dbadac852fca293f306b1791aac4f8e01cb37          
                    ea682485a644ee692aa58b'                                     
           INFO     checkpoint @                                 __main__.py:220
                    /home/undef13/.cache/splifft/bs_roformer-fruit-sw                
                    /model.ckpt (350.4 M)                                       
           INFO     config @                                     __main__.py:224
                    /home/undef13/.cache/splifft/bs_roformer-fruit-sw                
                    /config.json
```

To view details about a model, use `splifft info bs_roformer-fruit-sw` (optionally pass in `-a` to view model architecture details)

### BS Roformer

This is an audio-to-audio spectrogram-masking model.

- [`config.model`][splifft.models.bs_roformer.BSRoformerParams]
- [`config.model_type = "bs_roformer" or "mel_roformer"`][splifft.config.Config.model_type]

In `splifft`, Mel Roformers falls under the BS Roformers categority, with the [`splifft.models.bs_roformer.MelBandsConfig`][] field present and set.

### MDX23C

- [`config.model`][splifft.models.mdx23c.MDX23CParams]
- [`config.model_type = "mdx23c"`][splifft.config.Config.model_type]

#### Visualizations

The following are quick comparisons for the quality of different models, as evaluated on MVSep.

<!--
https://github.com/undef13/splifft/blob/main/scripts/mvsep.py
uv run scripts/mvsep.py correlations --instruments instrum --id 7534 --id 7573 --id 7768 --id 8257 --id 8303 --id 8393 --id 8362 --id 9505 --id 9475 --id 9580
uv run scripts/mvsep.py correlations --instruments vocals --id 7475 --id 7706 --id 8093 --id 8265 --id 8337 --id 8377 --id 9470
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
