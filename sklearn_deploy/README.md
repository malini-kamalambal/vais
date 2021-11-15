#### Training - Create the deployment pipeline

 gcloud builds submit --config cloudbuild.yaml .


training_code_change

training_new_input_dataset
1. Create pub sub topic training-new-input-dataset
2. Create subscription 


## Code update in cloud source repo.

## New input data trigger

1. Create topic and notification for changes in input dataset bucket.
Topic name: projects/virtual-anomaly/topics/training-new-input-dataset
Bucket Name: training-input-dataset

eg. gsutil notification create -t training-new-input-dataset -f json -e OBJECT_FINALIZE gs://training-input-dataset


2. Create cloud build trigger to start on new objects in the training-input-dataset bucket.


## On demand run
1. Create a new trigger with type webhook
2. Call webhook url in the format 

eg. curl -X POST -H "application/json" "https://cloudbuild.googleapis.com/v1/projects/virtual-anomaly/triggers/training-on-demand:yZZZZZZZZ--hQ&secret=ZZZZZ" -d "{}"


#### Prediction - Create the deployment pipeline

# on-demand-run

 curl -X POST -H "application/json" "https://cloudbuild.googleapis.com/v1/projects/virtual-anomaly/triggers/prediction-on-demand:webhook?key=XXXXX&secret=XXXX" -d "{}"
