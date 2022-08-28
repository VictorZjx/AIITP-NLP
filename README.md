# How to run

Create the Container:
Change to the directory containing Dockerfile and requirements.txt

docker build . -f Dockerfile -t <container name>

Run the container: 
docker run -it --rm -p XXXX:8888 -p YYYY:8501 -v ~/<directory of where the app.py is>:/workspace <container name>

For jupyter lab:
Tunnel XXXX then run the command: jupyter lab

for streamlit:
Tunnel YYYY then run the command: streamlit run app.py (Run the python notebooks first to generate the h5 file before running the app.py)

streamlit run app.py

XXXX for jupyter lab port
YYYY for streamlit port
