# Awareness Perturbation Complexity index
Development of an index for assessing the level of consciousness of healthy and disorder of consciousness individuals.

## Why is the documentation here?
The documentation was in the wiki page of Github, I (yacine mahdid) decided to switch to an in-code documentation for two reasons:
1. To not be tied to Github for documenting and reporting
2. To be able to version control the documentation page

I'm open to other suggestions to document the code and the analysis, however due to the highly changing nature of the analysis it is difficult to keep the documentation up-to-date the further away from the code it lies.

## Structure of the codebase
This repository is structured into projects/milestone/experiments.
- Project are defined as overarching objectives that needs many iteration to get right. They are vague objectives that usually lead to a paper publication. For example `create a quantitative index of dynamic reconfiguration`.
- Milestone is a step into achieving this objective without being as abstract as a project. For instance `create_rough_dpli_contrast_indexes`is a milestone in the objective to create a quantitative index of dynamic reconfiguration. One milestone require many experiments to get done.
- Experiment is a single piece of work that can be executed. In this repository the experiment are tied to a single task on Github. We have a naming convention that reflect which task it is addressing. This means we have `ex_XX_` where XX means a task number on github. 

The documentation for this repository is located inside the codebase. You can browse over here the different project/milestone docs.
- [Create a Quantitative Index of Dynamic Reconfiguration](./projects/create_a_quantitative_index_of_dynamic_reconfiguration/README.md)
	- [Create Rough dPLI Contrast Indexes](./projects/create_a_quantitative_index_of_dynamic_reconfiguration/create_rough_dpli_contrast_indexes/README.md)
	- [Augment DRI Power by Including Hub and Better dPLI](./projects/create_a_quantitative_index_of_dynamic_reconfiguration/augment_dri_power_by_including_hub_and_better_dpli/README.md)


