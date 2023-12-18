# Launch a collection of pipelines for different models in Valohai

This demo shows how to easily
- Run hyperparameter tuning as a part of a pipeline
- Scale pipelines to train the same model on different datasets

## When to use this demo
The customer wants to:
- Retrain their model for different contexts (multiple patients, devices, vehicles..)
    - "We have a model built, that is trained for device xyz123, but now we have to fit the model for 100s of other devices"
- Run parallel steps, e.g. hyperparameter tuning, inside a pipeline
- Train the model for different environments (e.g. edge, online, mobile...) or compare multiple variations of the model in a pipeline

## When not to use this demo
The customer:
- Has heavy focus on exploration using notebooks and doesn't use an IDE or benefit from dividing their work into separate pure Python scripts that could be combined into a pipeline.
- Doesn't need to take their work into production or cannot define who/what will use their model.

## Video

*video content*

## How to demo?
- Start by showing a completed pipeline and talking about the fundamentals.
- Create a new pipeline with the train node converted to a set of parallel steps in a pipeline (i.e. a Task but let's avoid Valohai jargon). 
    - Show how to change parameters, input dataset, environment...
    - Reuse the preprocess node from an earlier pipeline as that will take some time to run. 
    - Mention that reusing the nodes is relevant mainly for dev and debugging.
- Launch the pipeline and let it run until the human approval.
- Go to the YAML file in Github and show the pipeline, mention that you will share it after the call. 
- Back in the UI, show how to create parallel pipelines by using the pipeline level parameters.
    - Change the value of the pipeline parameter called "dataset". Available values:
        - train (default)
        - train_harbor_A
        - train_harbor_B
        - train_harbor_C
- Show how to create a trigger to run the pipeline for production runs (hourly, daily, monthly, etc.)


## Acknowledgements
The example here is based on the Ship classification Notebook by [Arpit Jain](https://www.kaggle.com/code/arpitjain007/ship-classification/notebook). 
- The dataset is available [here](https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets).