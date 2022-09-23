build:
	printenv GCP_KEYFILE > /home/vscode/keyfile.json
	pip install --upgrade pip wheel
	pip install -r requirements.txt -r requirements-dev.txt