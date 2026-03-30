
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

sphinx-build . _build

cd _build
python3 -m http.server 8102

If running on iynx, forward port
    ssh -N -L 8102:localhost:8102 zl@iynx

