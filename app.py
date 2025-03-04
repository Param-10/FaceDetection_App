#!/usr/bin/env python
"""
Simple entry point for the Face Detection Web App.
This file is placed at the root directory for easier execution.
"""
from FaceDetection_WebApp.app import app
import os

if __name__ == '__main__':
    print("Starting Face Detection Web App...")
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    # Use 0.0.0.0 to bind to all interfaces
    app.run(host='0.0.0.0', port=port, debug=False)
