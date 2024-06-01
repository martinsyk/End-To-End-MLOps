   1. Clone the directory to your system: "git clone https://github.com/martinsyk/End-To-End-MLOps.git!
   2. Navigate to the cloned project directory in your system: "cd [project_directory]"
   3. Create a virtual environment: "Python3 -m venv .venv"
   4. Activate the virtual environment: ".venv\Scripts\activate"
   4.1 In case of an error: "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
   5. Install the needed libraries: "pip install -r requirements.txt"
   6. Start the MFlow UI: "Mlflow ui"
   6.1 Based on terminal output open the link (it should be hosted on "selfhost:5000")
   7. Trigger DVC: "dvc init" (if already activated, ingore the error and continue)
   8. Initiate the pipeline: "dvc repro"
   9. Start the Evidently AI dashboard: "evidently ui --workspace ./workspace --port 8080"
   9.1 It should be availableon: http://localhost:8080/
   10. Run the prediction service: "python app.py"
   10.1 It should be available on "http://localhost/"

Deployment of the prediction service on Azure with Docker:
   1. Create azure Container registry
   2. Resource group
   3. Create Web App
   4. Create deployment  through Github Actions

