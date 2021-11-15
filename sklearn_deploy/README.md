## Training

### Creating pipeline

For the pipeline we are using Cloud Build.
The sklearn_deploy/training directory has the configuration file (cloudbuild.yaml) defining cloud build steps.

You can manually trigger a build with the command below,

```gcloud builds submit --config cloudbuild.yaml .```

#### Using cloud build triggers

A build can be triggered by various events such as code push to the code repository, a new training
dataset becoming available at cloud storage, due to a scheduled run etc.
See [documentation](https://cloud.google.com/build/docs/automating-builds/create-manage-triggers)
for additional details on cloud build triggers. We will discuss a few options below.

##### Trigger via code change
- Connect the repository from the Cloud Build page in Google Cloud Console.
- Create a new trigger for github events

##### Trigger via webhook
- Connect the repository from the Cloud Build page in Google Cloud Console.
- Create a new cloud build trigger for github events
- Use the preview url available in the triggers page to start the build to make HTTP POST calls. 
```eg. curl -X POST -H "application/json" "https://cloudbuild.googleapis.com/v1/projects/virtual-anomaly/triggers/prediction-on-demand:webhook?key=XXXXX&secret=XXXX" -d "{}"```


##### Trigger when a new input dataset or model is available in cloud storage.

- Set up pubsub notification for the cloud storage bucket. See [documentation](https://cloud.google.com/storage/docs/reporting-changes) for how to.
- Create a new cloud build trigger for the notifications.


## Prediction

Prediction pipeline is similar to the training pipeline we set up for training.
See sklearn_deploy/prediction/cloudbuild.yaml for a sample build configuration file.
The triggers are similar to training. Please follow the steps in the training section to
create triggers.

## Approving builds before executing
Cloud Build enables you to configure triggers that do not immediately execute a build but instead mark a build as pending until approved. If a user with permissions approves a pending build, the build will start. If the approval is denied, the build will not start. See [documentation](https://cloud.google.com/build/docs/automating-builds/approve-builds) on how to
create builds that require approval.
