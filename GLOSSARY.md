# Stack Overflow Tag Prediction

Proof of concept that suggests Stack Overflow tags for a question using only its title, comparing a classic machine-learning approach with a deep-learning approach.

## Data

**Post**:
A single Stack Overflow question, identified by its post id, with one title and one to five tags.
_Avoid_: Question, row, sample

**Title**:
The one-line text of a post; the only input the models see.
_Avoid_: Question, text, body

**Tag**:
A topic keyword attached to a post by its author (e.g. `c#`, `jquery`).
_Avoid_: Label (when talking about the raw data), category, class

**Tag position**:
The order in which a tag appears on a post. Carries no meaning of "main" or "primary" tag in this project.
_Avoid_: Primary tag, main tag

## Prediction

**Kept tag**:
A tag frequent enough in the dataset (at least 20 posts) to be predicted. Tags below the bar are ignored.
_Avoid_: Class, target

**Label set**:
The kept tags of a post: what a model must predict for that post. A post whose label set is empty is excluded.
_Avoid_: Ground truth, target tags

**Tag score**:
A model's confidence, between 0 and 1, that a given kept tag belongs to a title.
_Avoid_: Probability, logit

**Predicted tags**:
The kept tags whose tag score reaches the decision threshold, always containing at least the best-scored tag.
_Avoid_: Prediction, output

**ML model**:
The classic machine-learning approach: word statistics fed to one linear classifier per kept tag.
_Avoid_: Baseline, classic model

**DL model**:
The deep-learning approach: a pre-trained language model fine-tuned to score every kept tag at once.
_Avoid_: Neural model, transformer, BERT
