# Object location task
Last edit: 08/13/2024

## Edit history
- 08/13/2024 by Alex He - generated experiment scripts on python 3.8 for PsychoPy 2024.2.1
- 08/13/2024 by Julia Glueck - corrected typos in instructions, added trial_type and correct_resp to output csv file
- 07/02/2024 by Alex He - created finalized first draft version

## Description
This is a short-term memory task using a two-alternative forced choice (2AFC) test on old/new recognition of visual picture stimuli. The behavioral paradigm is adapted from a long history of object-location memory tasks:

Mitchell, K. J., Johnson, M. K., Raye, C. L., & D’Esposito, M. (2000). fMRI evidence of age-related hippocampal dysfunction in feature binding in working memory. Cognitive brain research, 10(1-2), 197-206.

Olson, I. R., Page, K., Moore, K. S., Chatterjee, A., & Verfaellie, M. (2006). Working memory for conjunctions relies on the medial temporal lobe. Journal of Neuroscience, 26(17), 4596-4601.

Das, S. R., Mancuso, L., Olson, I. R., Arnold, S. E., & Wolk, D. A. (2016). Short-term memory depends on dissociable medial temporal lobe regions in amnestic mild cognitive impairment. Cerebral Cortex, 26(5), 2006-2017.

In this task, subjects are asked to encode a series of three visual objects using a colored version of Snodgrass and Vanderwart's picture set from:

Rossion, B., & Pourtois, G. (2004). Revisiting Snodgrass and Vanderwart's object pictorial set: The role of surface detail in basic-level object recognition. Perception, 33(2), 217-236.

In each trial, three visual objects are presented one at a time for 1 second, and followed by an 8-second delay. After the delay, a prompt screen shows up stating 'Remember Object', 'Remember Location', or 'Remember Object and Location'. These prompt screens determine the type of the current trial to require single feature recognition or recognition of conjunctive features (in this case, object identity and location) in the encoded visual objects. As in the Das et al. 2016 paper, there are 24 trials per trial type, and 12 of them are in the old condition while the rest 12 are new.

N.B.: this paradigm is optimized for eliciting behavioral performance that could detect differences in short-term memory performance between the single feature condition (48 trials in total, 24 objects only and 24 locations only) versus the conjunction feature (24 conjunction trials) condition. This design is not ideal for studying neural signals during conjunctive encoding and retrieval after a short delay, because there are only 12 old conjunction trials that get tested. It is also difficult to analyze within-subject variance due to the low number of trials per subject.

## Outcome measures
- Behavioral performance in each condition (d')
- Alpha activity during the 8-s delay period
- A sharp alpha-beta power reduction at the onset of the prompt screen
- Potential parietal ERP old/new effect during testing
- (Unlikely) subsequent memory effects during encoding
