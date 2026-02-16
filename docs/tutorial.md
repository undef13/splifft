## Basic inference

Use the [`splifft.inference.InferenceEngine.from_pretrained`][] for a convenient high level API.

=== "inference.py"

    ```py
    --8<-- "docs/examples/inference.py:2"
    ```

=== "Output"

    ```txt
    [00:00:36] Stage.Started(stage='normalize', total_batches=None)
    [00:00:36] Stage.Completed(stage='normalize')
    [00:00:38] ChunkProcessed(batch_index=1, total_batches=9)
    [00:00:39] ChunkProcessed(batch_index=2, total_batches=9)
    [00:00:41] ChunkProcessed(batch_index=3, total_batches=9)
    [00:00:42] ChunkProcessed(batch_index=4, total_batches=9)
    [00:00:43] ChunkProcessed(batch_index=5, total_batches=9)
    [00:00:44] ChunkProcessed(batch_index=6, total_batches=9)
    [00:00:46] ChunkProcessed(batch_index=7, total_batches=9)
    [00:00:47] ChunkProcessed(batch_index=8, total_batches=9)
    [00:00:47] ChunkProcessed(batch_index=9, total_batches=9)
    [00:00:47] Stage.Started(stage='stitch', total_batches=None)
    [00:00:47] Stage.Completed(stage='stitch')
    [00:00:47] Stage.Started(stage='collect_outputs', total_batches=None)
    [00:00:47] Stage.Completed(stage='collect_outputs')
    [00:00:47] Stage.Started(stage='derive_stems', total_batches=None)
    [00:00:47] Stage.Completed(stage='derive_stems')
    InferenceOutput(
        outputs={
            'bass': tensor([[-1.3643e-05, -1.3736e-05, -1.3643e-05,  ..., 
    -1.3958e-05,
            -1.3730e-05, -1.3960e-05],
            [-1.3811e-05, -1.3586e-05, -1.3811e-05,  ..., -1.3738e-05,
            -1.3953e-05, -1.3736e-05]], device='cuda:0'),
            'drums': tensor([[-1.3493e-05, -1.4200e-05, -1.3493e-05,  ..., 
    -1.2080e-05,
            -1.2848e-05, -1.2020e-05],
            [-1.3936e-05, -1.3758e-05, -1.3936e-05,  ..., -1.1843e-05,
            -1.2818e-05, -1.1989e-05]], device='cuda:0'),
            'other': tensor([[-7.5168e-07, -6.3413e-07, -7.5222e-07,  ...,  
    1.9690e-05,
            -3.3400e-05,  2.5086e-05],
            [-7.4173e-07, -6.7063e-07, -7.4244e-07,  ...,  3.2220e-05,
            -3.7293e-05,  2.0826e-05]], device='cuda:0'),
            'vocals': tensor([[-1.3789e-05, -1.3904e-05, -1.3789e-05,  ..., 
    -1.3930e-05,
            -1.3755e-05, -1.4037e-05],
            [-1.3860e-05, -1.3833e-05, -1.3860e-05,  ..., -1.3848e-05,
            -1.3747e-05, -1.3913e-05]], device='cuda:0'),
            'guitar': tensor([[-1.3846e-05, -1.3846e-05, -1.3846e-05,  ..., 
    -1.3928e-05,
            -1.3760e-05, -1.3928e-05],
            [-1.3910e-05, -1.3782e-05, -1.3910e-05,  ..., -1.3871e-05,
            -1.3818e-05, -1.3871e-05]], device='cuda:0'),
            'piano': tensor([[-1.3789e-05, -1.3902e-05, -1.3789e-05,  ..., 
    -1.3933e-05,
            -1.3759e-05, -1.3933e-05],
            [-1.3881e-05, -1.3810e-05, -1.3881e-05,  ..., -1.3849e-05,
            -1.3843e-05, -1.3849e-05]], device='cuda:0'),
            'instrum': tensor([[ 1.3789e-05,  1.3904e-05,  1.3789e-05,  ...,  
    4.7834e-05,
            -2.5873e-05,  5.2345e-05],
            [ 1.3860e-05,  1.3833e-05,  1.3860e-05,  ...,  6.8972e-05,
            -3.0868e-05,  4.3241e-05]], device='cuda:0')
        },
        sample_rate=44100
    )
    ```

This outputs [`splifft.inference.InferenceOutput`][], containing:

- the dictionary of stem names to tensor (which can be audio or logits)
- the [sample rate][splifft.types.SampleRate] of the input tensor (so you can save the audio)

## Low level inference

This gives you more control:

```py title="inference_low_level.py"
--8<-- "docs/examples/inference_low_level.py"
```

## Extending `splifft`

`splifft` is designed to be easily extended without modifying its core.

Make sure you have [added `splifft` as a dependency](./index.md#library). Assuming your library has [this structure](https://github.com/undef13/splifft/tree/main/docs/examples/ext_project):

``` title="tree /path/to/ext_project"
├── pyproject.toml
├── scripts
│   └── main.py
└── src
    └── my_library
        └── models
            ├── __init__.py
            └── my_model.py
```

### 1. Define a new model

!!! warning "Don't do this"

    A common pattern is to define a model with a huge list of parameters in its `__init__` method:

    ```py title="src/my_library/models/my_model.py"
    from torch import nn
    from beartype import beartype

    class MyModel(nn.Module):
        @beartype
        def __init__(
            self,
            chunk_size: int,
            output_stem_names: tuple[str, ...],
            # ... a bunch of args here
        ):
            ...
    ```

    The problem is that it tightly couples the model's *implementation* to its *configuration*. Serializing to/from a JSON file and simultaneously supporting static type checking is a headache.

Instead, define a [stdlib `dataclass`][dataclasses.dataclass] separate from the model:

```py title="src/my_library/models/my_model.py" hl_lines="10"
--8<-- "docs/examples/ext_project/src/my_library/models/my_model.py"
```

1. [`ModelParamsLike`][splifft.models.ModelParamsLike] is *not* a base class to inherit from, but rather a form of [structural typing][typing.Protocol] that signals that `MyModelParams` is compatible with the [`splifft` configuration system][splifft.config.LazyModelConfig]. You can remove it if you don't like it.

### 2. Register the model

With the model and its config defined, our [configuration system][splifft.config.Config] needs to understand your model.

!!! warning "Don't do this"

    A common solution is to define a "global" dictionary of available models:

    ```py title="src/my_library/models/__init__.py"
    from my_library.models.my_model import MyModelParams, MyModel

    MODEL_REGISTRY = {
        "my_model": (MyModel, MyModelParams),
        # every other model must be added here
    }
    ```

    To add a new model, you'd have to modify this central registry. It also forces the import of all models and unwanted dependencies at once.

Instead, our [configuration system][splifft.config.Config] uses a simple [`ModelMetadata`][splifft.models.ModelMetadata] wrapper struct to act as a "descriptor" for your model. Create a factory function that defers the imports until its actually needed:

```py title="src/my_library/models/__init__.py"
--8<-- "docs/examples/ext_project/src/my_library/models/__init__.py:2:"
```

??? question "I need to take a user's input string and dynamically import the model. How?"

    [`ModelMetadata.from_module`][splifft.models.ModelMetadata.from_module] is an alternative way to load the model metadata. It uses [importlib][] under the hood. In fact, our CLI uses this exact approach.

    ```py
    from splifft.models import ModelMetadata

    my_model_metadata = ModelMetadata.from_module(
        module_name="my_library.models.my_model",
        model_cls_name="MyModel",
        model_type="my_model"
    )
    ```

### 3. Putting everything together

First, load in the [configuration](./models.md):

```py title="scripts/main.py"
--8<-- "docs/examples/ext_project/scripts/main.py:config"
```

This validates your JSON and returns a [pydantic.BaseModel][]. Note that at this point, [`config.model`][splifft.config.Config.model] is a [*lazy* model configuration][splifft.config.LazyModelConfig] that is not yet fully validated.

Next, we need to create the PyTorch model. Concretize the lazy model configuration into the `dataclass` we defined [earlier](#1-define-a-new-model) then instantiate the model:

```py title="scripts/main.py"
--8<-- "docs/examples/ext_project/scripts/main.py:model"
```

Finally, load the weights, input audio and run!

```py title="scripts/main.py"
--8<-- "docs/examples/ext_project/scripts/main.py:inference"
```
