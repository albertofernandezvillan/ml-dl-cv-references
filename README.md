# Machine Learning, Deep Learning and Computer Vision references
These are my quick and dirty notes trying to follow main Machine Learning, Computer Vision &amp; Deep Learning references

# Pytorch

* **the-incredible-pytorch**: The Incredible PyTorch: a curated list of tutorials, papers, projects, communities and more relating to PyTorch. https://github.com/ritchieng/the-incredible-pytorch

* **pytorch-Deep-Learning**: Deep Learning (with PyTorch). https://github.com/Atcold/pytorch-Deep-Learning
  * Website: https://atcold.github.io/pytorch-Deep-Learning/ 


## Pytorch Lightning
Pytorch Lightning main repo: https://github.com/PyTorchLightning 
*  **pytorch-lightning**: The lightweight PyTorch wrapper for high-performance AI research. https://github.com/PyTorchLightning/pytorch-lightning
*  **lightning-transformers**: Flexible interface for high performance research using SOTA Transformers leveraging Pytorch Lightning, Transformers, and Hydra.https://github.com/PyTorchLightning/lightning-transformers
*  **lightning-flash**: Collection of tasks for fast prototyping, baselining, finetuning and solving problems with deep learning. https://github.com/PyTorchLightning/lightning-flash
   *  Example (1): Image Classification using Lightning Flash. https://medium.com/nerd-for-tech/image-classification-using-lightning-flash-e549b6c4285f

## YOLO

*  **Complex-YOLOv4-Pytorch**: The PyTorch Implementation based on YOLOv4 of the paper: Complex-YOLO: Real-time 3D Object Detection on Point Clouds. https://github.com/maudzung/Complex-YOLOv4-Pytorch

## facebookresearch

*  **pytorchvideo**: A deep learning library for video understanding research. PyTorchVideo is developed using PyTorch and supports different deeplearning video components like video models, video datasets, and video-specific transforms. https://github.com/facebookresearch/pytorchvideo
* **detectron2**: Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. https://github.com/facebookresearch/detectron2
* **mobile-vision**: Mobile Computer Vision @ Facebook. https://github.com/facebookresearch/mobile-vision
* **hydra**: A framework for elegantly configuring complex applications. https://github.com/facebookresearch/hydra


# TensorFlow and Keras

* **Model Garden for TensorFlow**: The TensorFlow Model Garden is a repository with a number of different implementations of state-of-the-art (SOTA) models and modeling solutions for TensorFlow users. https://github.com/tensorflow/models
  * Example (1): Training Faster R-CNN Using TensorFlow’s Object Detection API with a Custom Dataset. https://pub.towardsai.net/training-faster-r-cnn-using-tensorflow-object-detection-api-with-a-custom-dataset-88dd525666fd
* **TensorFlow-Examples**: TensorFlow Tutorial and Examples for Beginners (support TF v1 & v2). https://github.com/aymericdamien/TensorFlow-Examples
* **Collection of tutorials**:
  * Autoencoder in TensorFlow 2: Beginner’s Guide:  https://learnopencv.com/autoencoder-in-tensorflow-2-beginners-guide/

# Optimization frameworks
* **optuna**: A hyperparameter optimization framework. https://github.com/optuna/optuna
   * Website: https://optuna.org/
   * Collection of unofficial optuna examples (scikit-learn, pytorch, tensorflow, xgboost,...) https://github.com/toshihikoyanase/optuna-examples



# Edge computing

* **tinyml-papers-and-projects**: This is a list of interesting papers and projects about TinyML. https://github.com/gigwegbe/tinyml-papers-and-projects


## Examples
* Deploy YOLOv5 to Jetson Xavier NX at 30FPS: https://blog.roboflow.com/deploy-yolov5-to-jetson-nx/. First, we have to train YOLO V5 (https://models.roboflow.com/object-detection/yolov5)

# Machine Learning

* **MadeWithML**: Learn how to responsibly deliver value with ML. https://github.com/GokuMohandas/MadeWithML
  * Website: https://madewithml.com/
  * Basics: https://github.com/GokuMohandas/madewithml/tree/main/notebooks
  * MLOps: https://github.com/GokuMohandas/mlops

* **applied-ml**: 📚 Papers & tech blogs by companies sharing their work on data science & machine learning in production. Curated papers, articles, and blogs on data science & machine learning in production. ⚙️. https://github.com/eugeneyan/applied-ml





* **PyCaret**: open-source, low-code machine learning library in Python that automates machine learning workflows. https://github.com/pycaret/pycaret
  *  Official Website: https://www.pycaret.org
  *  Documentation: https://pycaret.readthedocs.io/en/latest/
  *  Examples: https://github.com/pycaret/pycaret/tree/master/examples
  *  Example (1): Deploy Machine Learning Pipeline on the cloud using Docker Container. https://towardsdatascience.com/deploy-machine-learning-pipeline-on-cloud-using-docker-container-bec64458dc01

* **Scikit-learn**: machine learning in Python. https://github.com/scikit-learn/scikit-learn
  *  Website: https://scikit-learn.org
  *  Example (1): Build and Run a Docker Container for your Machine Learning Model. https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f


## Time series analysis
*  **statsmodels**: statistical models, hypothesis tests, and data exploration. https://github.com/statsmodels/statsmodels. **Example of use**: *Time Series Decomposition In Python (Time Series Analysis Made Easy)*: https://towardsdatascience.com/time-series-decomposition-in-python-8acac385a5b2 
*  Multiple Time Series Forecasting with PyCaret: https://towardsdatascience.com/multiple-time-series-forecasting-with-pycaret-bc0a779a22fe

## Machine learning examples
* 130 Machine Learning Projects Solved and Explained: https://medium.com/the-innovation/130-machine-learning-projects-solved-and-explained-605d188fb392

## Machine learning EDA
### Some examples:
* Titanic: Machine Learning from Disaster - EDA: https://www.kaggle.com/dwin183287/titanic-machine-learning-from-disaster-eda

## Outlier detection:
* 4 Machine learning techniques for outlier detection in Python. https://towardsdatascience.com/4-machine-learning-techniques-for-outlier-detection-in-python-21e9cfacb81d


# Exposing & deploying your applications

* **gradio**: Create UIs for prototyping your machine learning model in 3 minutes. Quickly create customizable UI components around your models. https://github.com/gradio-app/gradio
  * Example (1): Supercharge Your Machine Learning Experiments with PyCaret and Gradio. https://towardsdatascience.com/supercharge-your-machine-learning-experiments-with-pycaret-and-gradio-5932c61f80d9
* **streamlit**. The fastest way to build and share data apps. https://github.com/streamlit/streamlit
  * Example (1): Deploying Your Machine Learning Apps in 2021 (Streamlit Sharing is here and is awesome). https://towardsdatascience.com/deploying-your-machine-learning-apps-in-2021-a3471c049507
  * Example (2): Diabetes Prediction App. https://github.com/arunnthevapalan/diabetes-prediction-app
* **fastapi**: FastAPI framework, high performance, easy to learn, fast to code, ready for production. https://github.com/tiangolo/fastapi
  * Website: https://fastapi.tiangolo.com/
  * Example (1): How to deploy Machine Learning models as a Microservice using FastAPI. https://towardsdatascience.com/how-to-deploy-machine-learning-models-as-a-microservice-using-fastapi-b3a6002768af


# MLOPS

* **awesome-mlops**: A curated list of references for MLOps. https://github.com/visenger/awesome-mlops
  *  Website: https://ml-ops.org/

## Examples:
* How I Build Machine Learning Apps in Hours: https://pub.towardsai.net/how-i-build-machine-learning-apps-in-hours-a1b1eaa642ed?gi=7cbefed10ea6
* How to Dockerize Any Machine Learning Application: https://towardsdatascience.com/how-to-dockerize-any-machine-learning-application-f78db654c601. https://www.kdnuggets.com/2021/04/dockerize-any-machine-learning-application.html
* Deploying Your Machine Learning Apps in 2021: https://towardsdatascience.com/deploying-your-machine-learning-apps-in-2021-a3471c049507

# Kaggle
* **Hidden Gems: A Collection of Underrated Notebooks**. https://www.kaggle.com/headsortails/hidden-gems-a-collection-of-underrated-notebooks/
* **Embedding and Linking Notebooks**. https://www.kaggle.com/product-feedback/230748

# Interesting users repositories
* Jupyter Notebooks and other material from tutorial sessions on Machine Learning, Data Science, and related. https://github.com/dennisbakhuis/Tutorials

# Books (mainly with code)
* **dlwpt-code**: Code for the book Deep Learning with PyTorch by Eli Stevens, Luca Antiga, and Thomas Viehmann. https://github.com/deep-learning-with-pytorch/dlwpt-code
  * Website:  https://www.manning.com/books/deep-learning-with-pytorch


* **Practical Data Analysis using Jupyter Notebook**. https://github.com/PacktPublishing/Practical-Data-Analysis-using-Jupyter-Notebook

# Dataset "helpers"

* **fiftyone**: The open-source tool for building high-quality datasets and computer vision models. https://github.com/voxel51/fiftyone
   * Website: https://voxel51.com/docs/fiftyone/
   * Example (1): Stop Wasting Time with PyTorch Datasets! (Guide to speeding up and streamlining your dataset workflows with FiftyOne). https://towardsdatascience.com/stop-wasting-time-with-pytorch-datasets-17cac2c22fa8



# XXX Days of Machine and Deep Learning Code
* **ThinamXx/300Days__MachineLearningDeepLearning**: Journey of 300DaysOfData in Machine Learning and Deep Learning. https://github.com/ThinamXx/300Days__MachineLearningDeepLearning
* **jelifysh/100daysofmlcode**: https://github.com/jelifysh/100daysofmlcode/

# Google Colab tutorials
Intro to pandas: https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb
TensorFlow Programming Concepts: https://colab.research.google.com/notebooks/mlcc/tensorflow_programming_concepts.ipynb
First Steps with TensorFlow: https://colab.research.google.com/notebooks/mlcc/first_steps_with_tensor_flow.ipynb
Intro to Neural Networks: https://colab.research.google.com/notebooks/mlcc/intro_to_neural_nets.ipynb
Intro to Sparse Data and Embeddings: https://colab.research.google.com/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb
Tensorflow with GPU: https://colab.research.google.com/notebooks/gpu.ipynb
TPUs in Colab: https://colab.research.google.com/notebooks/tpu.ipynb
Retraining an Image Classifier (using TF Hub): https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb
Text Classification with Movie Reviews (using TF Hub): https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb

# Specific applications
## Medical applications
* Open Source project to help predict Tuberculosis from chest X-Rays (Tensorflow, OpenCV, Scikit-Learn). https://github.com/darwinai/TuberculosisNet
* 
# Cheat Sheets

* Your Ultimate Data Mining & Machine Learning Cheat Sheet (using scikit-learn). https://towardsdatascience.com/your-ultimate-data-mining-machine-learning-cheat-sheet-9fce3fa16

# Quick and dirty notes (just for me)

* How to Choose a CI Framework for Deep Learning: https://medium.com/pytorch-lightning/how-to-choose-a-ci-framework-for-deep-learning-d24ee9ef902c
* Data Science 101: Normalization, Standardization, and Regularization: https://www.kdnuggets.com/2021/04/data-science-101-normalization-standardization-regularization.html
* How to Combine Predictions for Ensemble Learning: https://machinelearningmastery.com/combine-predictions-for-ensemble-learning/. Also this one: https://towardsdatascience.com/ensemble-learning-bagging-boosting-3098079e5422 and this one: https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
* A Layman’s guide to ROC Curves And AUC. https://mlwhiz.com/blog/2021/02/03/roc-auc-curves-explained/?utm_campaign=a-laymans-guide-to-roc-curves-and-auc&utm_medium=social_link&utm_source=missinglettr
* Explainable AI (XAI) with a Decision Tree (Practical guide for XAI analysis with Decision Tree visualization): https://towardsdatascience.com/explainable-ai-xai-with-a-decision-tree-960d60b240bd
* Data Cleaning: Hidden Aspect of Good Data Scientist: https://dockship.io/articles/60748d13dba94b312797af1e/data-cleaning:-hidden-aspect-of-good-data-scientist
* How to Use Variance Thresholding For Robust Feature Selection: https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f
* How to Use Pairwise Correlation For Robust Feature Selection: https://towardsdatascience.com/how-to-use-pairwise-correlation-for-robust-feature-selection-20a60ef7d10
* Comprehensive data exploration with Python: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
* Plant Pathology with Lightning: https://www.kaggle.com/jirkaborovec/plant-pathology-with-lightning
* Scaled-YOLOv4: https://blog.roboflow.com/scaled-yolov4-tops-efficientdet/
* Fusing EfficientNet & YoloV5: https://towardsdatascience.com/fusing-efficientnet-yolov5-advanced-object-detection-2-stage-pipeline-tutorial-da3a77b118d1
* A journey of building an Advanced Object Detection Pipeline — Doubling YoloV5’s performance: https://towardsdatascience.com/a-journey-of-building-an-advanced-object-detection-pipeline-doubling-yolov5s-performance-b3f1559463bf
* Exploratory Data Analysis, Visualization, and Prediction Model in Python: https://towardsdatascience.com/exploratory-data-analysis-visualization-and-prediction-model-in-python-241b954e1731
* How to Choose an Activation Function for Deep Learning. https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
* FLIR_to_Yolo. This script converts FLIR thermal dataset annotations to YOLO format. https://github.com/albertofernandezvillan/FLIR_to_Yolo
* Dimensionality reduction with Autoencoders versus PCA. https://towardsdatascience.com/dimensionality-reduction-with-autoencoders-versus-pca-f47666f80743
* P-value Explained Simply for Data Scientists: https://mlwhiz.com/blog/2019/11/11/pval/?utm_campaign=p-value-explained-simply-for-data-scientists&utm_medium=social_link&utm_source=missinglettr-linkedin

