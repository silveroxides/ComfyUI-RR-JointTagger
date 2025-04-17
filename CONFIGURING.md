# Configuratiion

## FDTagger: Node Documentation

This section provides an explanation of the input and output nodes for the FDTagger widget.

### Adding the node

1.  Add the node via `Furry Diffusion` -> `FDTagger`.
2.  Connect your inputs, adjust your parameters, and connect your outputs.
3.  Iteratively make changes as you go! 

	> 🐺 Models are automatically downloaded at runtime if missing.

The node supports tagging and outputting multiple batched inputs.  It can handle single images, or batched image operation input.

### Connecting the node

### Inputs

| **Input** | **Description** |
| --- | ----------------------- |
| ``image[]`` | An image or a batch of images to interrogate |

### Parameters

| **Parameter** | **Description** |
| --- | ----------------------- |
| ``model`` | The interrogation model to use for e621 tag interrogation |
| ``device`` | The device to use.  Select "cpu" or "cuda" from the menu |
| ``steps`` | This parameter does nothing for now |
| ``threshold`` | The score for the tag to be considered valid |
| ``replace_underscore`` | Set to **true** or check the box to replace underscores in tags with spaces |
| ``trailing_comma`` | Add a trailing comma to the caption output |
| ``exclude_tags`` | A comma separated list of tags that should not be included in the results |

### Outputs

| **Output** | **Description** |
| --- | ----------------------- |
| ``tags`` | A string array of prompts, in sequential order of image batch (if multiple)
The input nodes for the FDTagger class define the necessary inputs required for the tagging process. |
| ``scores`` | A list of tags and parired scores, this is the result of classification. |

------

## Setting up your `config.json`

This section provides a detailed explanation of the `config.json` file used for configuring the FurryDiffusion FDTagger extension.

This configuration file is essential for setting up the FurryDiffusion FDTagger extension, specifying how models and tags are managed, and defining default settings for the tagging process.

### General Settings

- **`name`**: Specifies the name of the extension.
  - Type: `string`
  - Example: `"furrydiffusion.FDTagger"`

- **`logging`**: Enables or disables logging.
  - Type: `boolean`
  - Example: `true`

- **`loglevel`**: Sets the logging level.
  - Type: `string`
  - Values: `"debug"`, `"info"`, `"warning"`, `"error"`
  - Example: `"debug"`

- **`huggingface_endpoint`**: Base URL for downloading models and tags from Hugging Face.
  - Type: `string`
  - Example: `"https://huggingface.co"`

- **`api_endpoint`**: Endpoint for the API.
  - Type: `string`
  - Example: `"furrydiffusion"`

### Image Cache Settings

- **`image_cache_maxsize`**: Maximum size of the image cache.
  - Type: `integer`
  - Example: `100`

- **`image_cache_method`**: Method used for image cache cleanup.
  - Type: `int`
  - Values: `"round_robin"`, `"least_recently_used"`
  - Example: `"round_robin"`

### FDTagger Settings

- **`fdtagger_settings`**: Contains settings specific to the FDTagger functionality.
  - **`model`**: Default model used for tagging.
    - Type: `string`
    - Example: `"RedRocket PILOT v1"`
  - **`steps`**: Number of steps for the tagging process.
    - Type: `int`
    - Example: `255`
  - **`threshold`**: Confidence threshold for tag inclusion.
    - Type: `float`
    - Example: `0.35`
  - **`exclude_tags`**: Tags to be excluded from the results.
    - Type: `string`
    - Example: `""` (empty string)
  - **`replace_underscore`**: Whether to replace underscores in tag names.
    - Type: `boolean`
    - Example: `true`
  - **`trailing_comma`**: Whether to include a trailing comma in the tag list.
    - Type: `boolean`
    - Example: `true`
  - **`device`**: Device used for computation.
    - Type: `string`
    - Values: `"cuda"`, `"cpu"`
    - Example: `"cuda"`

### Models Configuration

- **`models`**: Dictionary containing the configurations for different models.
  - **`<model_id>`**: Each key is a model identifier, and the value is a dictionary with model details.
    - **`name`**: Display name of the model.
      - Type: `string`
      - Example: `"RedRocket PILOT v1"`
    - **`url`**: URL template for downloading the model.
      - Type: `string`
      - Example: `"{HF_ENDPOINT}/RedRocket/JointTaggerProject/resolve/main/JTP_PILOT"`
    - **`version`**: Version of the model.
      - Type: `string`
      - Example: `"1"`

### Tags Configuration

- **`tags`**: Dictionary containing the configurations for different tag sets.
  - **`<tag_id>`**: Each key is a tag set identifier, and the value is a dictionary with tag details.
    - **`name`**: Display name of the tag set.
      - Type: `string`
      - Example: `"RedRocket PILOT v1"`
    - **`url`**: URL template for downloading the tag set.
      - Type: `string`
      - Example: `"{HF_ENDPOINT}/RedRocket/JointTaggerProject/resolve/main/JTP_PILOT"`
    - **`version`**: Version of the tag set.
      - Type: `string`
      - Example: `"1"`

### Example ``config.json``

Here is the provided `config.json` file as an example:

```json
{
  "name": "furrydiffusion.FDTagger",
  "logging": true,
  "loglevel": "debug",
  "huggingface_endpoint": "https://huggingface.co",
  "api_endpoint": "furrydiffusion",
  "image_cache_maxsize": 100,
  "image_cache_method": "round_robin",
  "fdtagger_settings": {
    "model": "RedRocket PILOT v1",
    "steps": 255,
    "threshold": 0.35,
    "exclude_tags": "",
    "replace_underscore": true,
    "trailing_comma": true,
    "device": "cuda"
  },
  "models": {
    "JTP_PILOT-e4-vit_so400m_patch14_siglip_384": {
      "name": "RedRocket PILOT v1",
      "url": "{HF_ENDPOINT}/RedRocket/JointTaggerProject/resolve/main/JTP_PILOT",
      "version": "1"
    },
    "JTP_PILOT2-e3-vit_so400m_patch14_siglip_384": {
      "name": "RedRocket PILOT v2",
      "url": "{HF_ENDPOINT}/RedRocket/JointTaggerProject/resolve/main/JTP_PILOT2",
      "version": "2"
    }
  },
  "tags": {
    "JTP_PILOT-e4-vit_so400m_patch14_siglip_384": {
      "name": "RedRocket PILOT v1",
      "url": "{HF_ENDPOINT}/RedRocket/JointTaggerProject/resolve/main/JTP_PILOT",
      "version": "1"
    },
    "JTP_PILOT2-e3-vit_so400m_patch14_siglip_384": {
      "name": "RedRocket PILOT v2",
      "url": "{HF_ENDPOINT}/RedRocket/JointTaggerProject/resolve/main/JTP_PILOT2",
      "version": "2"
    }
  }
}
```