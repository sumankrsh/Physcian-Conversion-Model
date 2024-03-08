# Physician/HCP Conversion MLOPs 

## Business Overview

**Introduction: Physician Conversion Model**

A Physician Conversion Model aims to predict the likelihood of a healthcare provider (HCP) becoming a writer for a particular company or brand for the first time. It helps pharmaceutical or medical device companies identify potential HCPs who have not yet written prescriptions for their products but may be receptive to doing so in the future. The model focuses on converting non-writing HCPs into active writers.

### Detailed View of an HCP Conversion Model

**Target Variable:** The target variable in an HCP conversion model is binary, indicating whether an HCP becomes a writer (1) or not (0) for a specific company or brand. This variable serves as the outcome to be predicted.

**Predictive Features:** Various data sources and types can be used to develop an HCP conversion model. Some common predictive features include:

- **Demographic Data:** Information about the HCP's age, gender, specialty, location, education level, and other relevant characteristics. Demographic data helps identify patterns and preferences specific to certain groups of HCPs.

- **Claims Data:** Historical claims data can provide insights into the HCP's prescribing behavior, utilization patterns, and therapeutic areas of interest. It helps identify HCPs with similar patient profiles or conditions that align with the company's products.

- **Promotional Data:** Data on past promotional activities directed at the HCP, such as detailing visits, samples provided, speaker programs attended, and engagement with educational materials. This helps assess the level of previous interaction and the impact of promotional efforts.

- **Network Analysis:** Examination of the HCP's professional network, affiliations, collaborations, and referral patterns. Understanding the relationships between HCPs can help identify opinion leaders and assess the impact of peer influence.

*Note: Other types of features can also be utilized depending on availability and granularity.*

## Project Details

### Content

The project is an implementation of MLOps capabilities using free sources. The breakdown is given in the following steps:

- For the ML model lifecycle, we have followed the FTI Pipeline structure (Feature Pipeline, Training Pipeline, and Inference Pipeline). For complete understanding, refer to the link: [FTI Pipelines](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines?utm_campaign=fti_pipelines_blog&utm_medium=email&_hsmi=275020632&_hsenc=p2ANqtz-9rHqy-8PLHpSG77WUG1j4SEu_I16iPhi61_9mPgJxYu72pZBxlXBAgWnpB52ZzZUkm8ENBxsGdhwuV0hD6SqRnIuStZ6X0NsdoM0ybtE_H9mcZ0Mo&utm_source=newsletter)

### Tools Used

- **Version Control and CI/CD:** GitHub & GitHub Actions
- **Feature Store:** Hopsworks (Free version)
- **Data Storage:** AWS S3 (trial version)
- **Model Tracking:** MLflow

### Folder Structure

The project follows a structured folder organization:

- **Data:** Contains input data and notebook output for reference. Actual data will be accessed from AWS S3.
- **Data Science:** Includes Jupyter notebooks in the subfolder "notebooks" for explainability and experimentation, as well as the "src" folder for Python files.
- **Conf:** Contains YAML files for reading parameters.

Feel free to explore the project folders and resources to better understand the implementation.
