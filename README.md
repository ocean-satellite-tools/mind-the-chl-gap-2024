# mind-the-chl-gap

Neural network models for Chloraphyll-a gap-filling for remote-sensing products

## Problem

Chlorophyll is a key indicator of marine productivity and the health of ocean ecosystems. By estimating chlorophyll levels, scientists gauge the abundance of phytoplankton, which in turn reflects the overall productivity and food availability for many types of forage fish. Remote-sensing via ocean-color sensors allow one to derive an index of concentration of chlorophyll-a in the water based on the reflectance characteristics of plankton and allows us to use remote-sensing to monitor chlorophyll on a global scale. However clouds, especially, are a major problem for ocean color instruments. Clouds obstruct the view of the ocean surface, leading to gaps in the chlorophyll-a data. Close to 70% of the earth is covered by clouds at any time. Other problems, like sun-glint, also cause gaps but clouds are a very pervasive cause of gaps.  Gaps in ocean color measurements significantly hinder our ability to monitor ocean productivity and to create ‘production’ level products to help fisheries. 

| <img width="906" alt="image" src="https://github.com/user-attachments/assets/f8a098a0-ef7a-447d-b468-f00954e95094"> | 
|:--:| 
| *Figure: Raw gappy chl-a estimates and gap-filled chl-a map.* |

Some parts of the world are more impacted by clouds than others. The Arabian Sea between East Africa and India is one of these regions. This region is home to two important and dynamic coast upwelling zones that drive large seasonal chlorophyll blooms that in turn drive the fisheries in this region and are critical for regional food security. The chlorophyll blooms happen during the summer monsoon when clouds are thick and winds drive coastal upwelling. Lack of data on chlorophyll intensity means poor spatial predictions of where fish, especially oil sardines, will be. Much of the fishing in this region is artisanal, on small boats that cannot go far. The Indian government produces fishing predictions based, in part, on Chl-a estimates. When it is cloudy, as it typically is during the monsoon, the Chl-a data are missing and fishing predictions are more uncertain. 

Convolutional neural networks are able to learn local patterns and features in images and use this information to create predictions. This type of model has been used successfully for other types of earth data gap-filling problems. Our work ([slideshow](https://docs.google.com/presentation/d/1etA0GIpuRJrahZnKkC36YrCN8A0dmnXPTETdv5v1m0E/edit?usp=sharing)) is testing a variety of different machine-learning models and comparing them to existing Chl-a gap-filling algorithms.  For UWHackWeek 2024, we would like to add a new, experimental, type of CNN,  physics informed CNNs (PINNs) and LSTM PINNs, to our set of models. We also want to add more tests, visualizations, and notebooks to help others apply PINNs to their work.

<img width="623" alt="image" src="https://github.com/user-attachments/assets/4ea38c95-67e9-44ed-939f-3fb8ac6bcf54">
Figure: Results from our U-Net model. Top left panel is the Copernicus Level-4 (science grade) Gap-filled Chl-a product and Top right panel is our U-Net gap-filled product using the Copernicus Level-3 (gappy) data plus co-located environmental variables.

## Files

* `.gitignore`
<br> Globally ignored files by `git` for the project.
* `environment.yml`
<br> `conda` environment description needed to run this project.
* `README.md`
<br> Description of the project (see suggested headings below)

## Folders

This template provides the following folders and suggested organizaiton structure for the project repository, but each project team is free to organize their repository as they see fit.

### `contributors`
Each team member can create their own folder under contributors, within which they can work on their own scripts, notebooks, and other files. Having a dedicated folder for each person helps to prevent conflicts when merging with the main branch. This is a good place for team members to start off exploring data and methods for the project.

### `notebooks`
Notebooks that are considered delivered results for the project should go in here.

### `scripts`
Helper utilities that are shared with the team should go in here.

# Recommended content for your README.md file:

## Project Summary

### Project Title

Brief title describing the proposed work.

### Collaborators

List all participants on the project.

* Project lead
* Team member
* Team member
* Team member
* ...

### The problem

What problem are you going to explore? Provide a few sentences. If this is a technical exploration of software or data science methods, explain why this work is important in a broader context and specific applications of this work. To get some ideas, see example use cases on the GeoSMART website [here](https://geo-smart.github.io/usecases).

### Specific questions / project goals

List the specific tasks you want to accomplish, project goals, or research questions you want to answer. Think about what outcomes or deliverables you'd like to create (e.g. a series of tutorial notebooks demonstrating a [use case](https://geo-smart.github.io/usecases#Contributing), or a new python package).

### Data

Briefly describe the data that will be used here (size, format, how to access).

### Existing methods

How would you or others traditionally try to address this problem?

### Proposed methods/tools

What new approaches would you like to try to implement to address your specific question(s) or application(s)?

### Additional resources or background reading

Optional: links to manuscripts or technical documents providing background information, context, or other relevant information.

### Tasks

What are the individual tasks or steps that need to be taken to achieve the project goals? Think about which tasks are dependent on prior tasks, or which tasks can be performed in parallel. This can help divide up project work among team members.

* Task 1 (all team members)
* Task 2
  * Task 2a (assigned to team member A)
  * Task 2b (assigned to team member B)
* Task 3
* ...
