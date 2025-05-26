SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
git clone https://github.com/Luxtelligence/lxt_pdk_gf.git
cd lxt_pdk_gf
pip install -e . pre-commit
pre-commit install
python install_tech.py