
####################
# Housekeeping
####################

# list just commands
default:
  just --list

install:
  pip install -U pip
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
  pip install -e .
  pip install opencv-python pycocotools matplotlib onnxruntime onnx jupyterlab fastapi uvicorn[standard] python-multipart shapely

jupyter PORT="8765":
  jupyter lab --no-browser --allow-root --port={{PORT}} --ip=0.0.0.0 .

app:
  uvicorn demo_app:app --reload --port 8081 --host 0.0.0.0
