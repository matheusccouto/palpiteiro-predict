build: pip
	printenv GCP_KEYFILE > /home/vscode/keyfile.json
	pip install -r requirements.txt -r requirements-train.txt -r requirements-test.txt
	echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
	curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
	sudo apt-get update && sudo apt-get install google-cloud-cli
pip:
	pip install --upgrade pip wheel
diagrams: pip
	sudo apt install graphviz -y
	pip install diagrams