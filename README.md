# ML-Image-Processor

My project idea is an image processing web application. Speaking from personal experience, the process of organizing photos is a huge pain. But when I don't organize them, I will have to go through the pain of searching through huge unsorted folders, looking for that one photo I knew I took. To reduce my pain, and the pain of photographers everywhere, I want to use machine learning to, among other things, automagically add tags to images.

First, a user of this application would upload the images that they want tagged. A model made using transfer learning would identify objects (i.e., banana, bear, tennis racket, face) in these images, which would be used to assign them category tags (i.e., nature, urban, group photo, portrait, food). Images with faces would be displayed on the next page. The user could then manually classify around 10 images of each person, which would be used to train another model to identify the remainder of the faces. The tagged images can then be displayed in order of the model's classification confidence, so that the user can easily edit incorrect tags. The final step would be to output the images saved with the new metadata in a zip folder.

Other tools that can be implemented (time-permitting, or if facial recognition is too inaccurate) include low-light enhancement and rotation correction. After conducting thorough research, I am confident that the web application I have outlined above can be developed by a group of motivated computer engineering students with a good understanding of CPEN 291 content and a desire to learn the basics of full-stack web development. I will describe the technical specifications of the proposed project in the following sections.

## Current State of Application

- Can upload and view images
- Uploaded images stored in the directory `uploads/`
![Application Screenshot](/Images/Application%20Screenshot.png)

## Dependencies

- [python3](https://www.python.org/download/releases/3.0/)
- [flask](https://flask.palletsprojects.com/en/1.1.x/installation/)

## How to Run

1. Run `python main.py` to start application
2. Click on the box to select which images to upload, or drag them to the box
![Select Images](/Images/Selecting%20Images.png)
