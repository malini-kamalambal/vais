#### Training - Create the deployment pipeline

 gcloud builds submit --config cloudbuild.yaml .


training_code_change

training_new_input_dataset
1. Create pub sub topic training-new-input-dataset
2. Create subscription 

#### Prediction - Create the deployment pipeline


1. Create topic and notification for changes in input dataset bucket
Topic name: projects/virtual-anomaly/topics/training-new-input-dataset
Bucket Name: training-input-dataset

gsutil notification create -t training-new-input-dataset -f json gs://training-input-dataset

