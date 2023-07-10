# In this project, we are going to write an example code for image classification.
# The project will firstly load images from the directory, apply hugging-face modules
# to build the model and preprocess the dataset. Then we will use transfer learning to
# finetune the model and apply Lora to do the PEFT. We will try to code using Hugging-face code,
# like Trainer rather than typical pytorch code.

# We might need to integrate both approach together. 
# Trainer works unexpected when using a script, we might need to add it as an option
