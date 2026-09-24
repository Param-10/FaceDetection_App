# Deploy on Render

This repository includes a Docker-based Render configuration. One public service serves both the Flask API and the compiled React frontend.

## Prerequisites

- Push the deployment changes to the repository's `main` branch.
- A Render account with access to the GitHub repository.
- The free plan is supported for testing, but it sleeps after inactivity and its 512 MB memory limit may be insufficient for TensorFlow/DeepFace.

## Create the service

1. In Render, choose **New → Blueprint**.
2. Connect `Param-10/FaceDetection_App`.
3. Select the branch containing `render.yaml`.
4. Review the `face-detection-app` service and choose a region.
5. Apply the Blueprint.

Render builds `Dockerfile`, starts Gunicorn on Render's `$PORT`, and checks `/health`. The frontend and API share the generated `onrender.com` URL.

The first startup may take several minutes while DeepFace downloads model weights. The public feedback database is created inside the container and is ephemeral unless a persistent disk is configured.

## If the Render service already exists

Do not create a second service. In the Render Dashboard, open the existing service and verify:

- Repository: `Param-10/FaceDetection_App`
- Branch: `main`
- Plan: Free
- Health check path: `/health`

For the existing native Python/free service, use:

- Build command: `pip install -r requirements.txt && npm ci && npm run build`
- Start command: `gunicorn app:app`

That lightweight mode runs OpenCV detection only. To enable DeepFace emotion, age, and gender analysis, use the Docker configuration and `requirements-deploy.txt` instead. After saving the settings, use **Manual Deploy → Deploy latest commit**.

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
