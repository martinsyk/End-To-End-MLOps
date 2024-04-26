   1. git clone https://github.com/martinsyk/End-To-End-MLOps.git
   2. cd [project_directory]
   3. Python3 -m venv .venv
   4. .venv\Scripts\activate
      a. Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser (if an error)
   5. pip install -r requirements.txt
   6. Mlflow ui
      a. Based on terminal output open the link or it Should be hosted on selfhost:5000
   7. Dvc init (might already be create, just error and continue)
   8. Dvc repro
   9. evidently ui --workspace ./workspace --port 8080
      a. http://localhost:8080/
   10. Python app.py
      a. http://localhost/


Deployment on azure with docker:
   1. Create azure Container registry
   2. Resource group
   3. Create Web App
   4. Create deployment through Github actions

