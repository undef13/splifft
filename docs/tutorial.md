## Basic inference

This example demonstrates the lower level API for inference usecases.
In the future, we will have a high level API for convenience.

```py title="inference.py"
--8<-- "docs/examples/inference.py"
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
