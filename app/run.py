import uvicorn
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

venv_site_packages_path = '/Users/kakhan/Documents/Coding_workspace/python/llm/llm-poc/.venv/lib/python3.12/site-packages'
sys.path.append(venv_site_packages_path)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
