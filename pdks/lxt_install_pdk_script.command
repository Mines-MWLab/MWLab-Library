SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
cd lxt_pdk_gf
pip install -e . pre-commit
pre-commit install
python3 install_tech.py