Setup instruction

Create a python 3 anaconda enviroment

conda install pybind11

git clone https://github.com/ConorFWild/clipper_python.git; cd clipper_python; python3 bundle_builder.py wheel; cd dist python3 -m pip install --user ./Clipper_Python-0.2*.whl

git clone https://github.com/ConorFWild/mdc3.git; cd mdc3; pip install .

git clone https://github.com/ConorFWild/pandda.git; cd pandda; pip install .
