
#!/bin/sh
apt update
apt-get install -y libgomp1
gunicorn -k uvicorn.workers.UvicornWorker source.app:app