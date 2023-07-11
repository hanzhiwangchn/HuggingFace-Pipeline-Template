# HuggingFace-Pipeline-Template
A template pipeline for Image Classification using HuggingFace library

In this project, we would like to provide an end-to-end pipeline using HuggingFace library. We use image classification as an example.

To preserve flexibility, we also provide a Pytorch pipeline for model training. Two pipelines could be used interchangeably to some degree. However, our models rely heavily on HuggingFace library, such as transformers and PEFT models.

Our pipeline contains several parts: Dataset Building, Model Building, Training Phase and testing.