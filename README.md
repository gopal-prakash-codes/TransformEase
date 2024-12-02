# Setup

### With Conda

2. Create a new virtual environment and install packages.

```bash
$ conda create -n st python pandas tqdm
$ conda activate st
```

Using Cuda:

```bash
$ conda install pytorch>=1.6 cudatoolkit=11.0 -c pytorch
```

Without using Cuda

```bash
$ conda install pytorch cpuonly -c pytorch
```

3. Install `simpletransformers`.

```bash
$ pip install simpletransformers
```

#### Optional

1. Install `Weights` and `Biases` (wandb) for tracking and visualizing training in a web browser.

```bash
$ pip install wandb
```

## Usage


`Simple Transformer` models are built with a particular Natural Language Processing (NLP) task in mind. Each such model comes equipped with features and functionality designed to best fit the task that they are intended to perform. The high-level process of using Simple Transformers models follows the same pattern.

1. Initialize a task-specific model
2. Train the model with `train_model()`
3. Evaluate the model with `eval_model()`
4. Make predictions on (unlabelled) data with `predict()`

However, there are necessary differences between the different models to ensure that they are well suited for their intended task. The key differences will typically be the differences in input/output data formats and any task specific features/configuration options. These can all be found in the documentation section for each task.

The currently implemented task-specific `Simple Transformer` models, along with their task, are given below.

| Task                                                      | Model                           |
| --------------------------------------------------------- | ------------------------------- |
| Binary and multi-class text classification                | `ClassificationModel`           |
| Conversational AI (chatbot training)                      | `ConvAIModel`                   |
| Language generation                                       | `LanguageGenerationModel`       |
| Language model training/fine-tuning                       | `LanguageModelingModel`         |
| Multi-label text classification                           | `MultiLabelClassificationModel` |
| Multi-modal classification (text and image data combined) | `MultiModalClassificationModel` |
| Named entity recognition                                  | `NERModel`                      |
| Question answering                                        | `QuestionAnsweringModel`        |
| Regression                                                | `ClassificationModel`           |
| Sentence-pair classification                              | `ClassificationModel`           |
| Text Representation Generation                            | `RepresentationModel`           |
| Document Retrieval                                        | `RetrievalModel`                |

### A quick example

```python
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    ["Aragorn was the heir of Isildur", 1],
    ["Frodo was the heir of Isildur", 0],
]
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]

# Preparing eval data
eval_data = [
    ["Theoden was the king of Rohan", 1],
    ["Merry was the king of Rohan", 0],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])

```


---

## Current Pretrained Models


The `model_types` available for each task can be found under their respective section. Any pretrained model of that type
found in the Hugging Face docs should work. To use any of them set the correct `model_type` and `model_name` in the `args`
dictionary.

*If you should be on this list but you aren't, or you are on the list but don't want to be, please don't hesitate to contact me!*

---

## How to Contribute

### How to Update Docs

below are the steps to edit the docs.
Docs are built using [Jekyll](https://jekyllrb.com/) library, refer to their webpage for a detailed explanation of how it works.

1) **Install [Jekyll](https://jekyllrb.com/)**: Run the command `gem install bundler jekyll`
2) **Visualizing the docs on your local computer**:
   In your terminal cd into the docs directory of this repo, eg: `cd simpletransformers/docs`
   From the docs directory run this command to serve the Jekyll docs locally: `bundle exec jekyll serve`
   Browse to http://localhost:4000 or whatever url you see in the console to visualize the docs.
3) **Edit and visualize changes**:
   All the section pages of our docs can be found under `docs/_docs` directory, you can edit any file you want by following the markdown format and visualize the changes after refreshing the browser tab.
