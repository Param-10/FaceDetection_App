# Deploy on Render

This repository includes a Docker-based Render configuration. One public service serves both the Flask API and the compiled React frontend.

## Prerequisites

- Push the deployment changes to the repository's `main` branch.
- A Render account with access to the GitHub repository.
- A paid `1c-2g` web-service plan for reliable TensorFlow/DeepFace performance. The free 512 MB tier is not recommended for this workload.

## Create the service

1. In Render, choose **New → Blueprint**.
2. Connect `Param-10/FaceDetection_App`.
3. Select the branch containing `render.yaml`.
4. Review the `face-detection-app` service and choose a region.
5. Apply the Blueprint.

Render builds `Dockerfile`, starts Gunicorn on Render's `$PORT`, and checks `/health`. The frontend and API share the generated `onrender.com` URL.

The first startup may take several minutes while DeepFace downloads model weights. The public feedback database is created inside the container and is ephemeral unless a persistent disk is configured.

## Optional persistent feedback storage

For a paid service, attach a disk mounted at `/var/data` and add:

```yaml
- key: MODEL_DATA_DIR
  value: /var/data/model_data
```

This preserves SQLite feedback events across restarts and deploys. Without it, the container filesystem is temporary.

## Local production-image check

If Docker is installed:

```bash
docker build -t face-detection-app .
docker run --rm -p 5050:10000 face-detection-app
```

Then open <http://localhost:5050> and check <http://localhost:5050/health>.

## Before exposing it to real users

The upload endpoint is rate-unlimited and has no authentication. Add authentication, rate limiting, a privacy notice, and a retention/deletion policy before accepting real face images. The image upload and decoded-pixel limits reduce resource-exhaustion risk but are not access controls.
