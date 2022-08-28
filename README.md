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


# Movie sentiment analysis demo:<br>

![Movie sentiment analysis demo](https://user-images.githubusercontent.com/87589755/187067578-10b823d4-b7ed-4430-a404-66d496445acf.gif)


# Translation + QnA demo:

![Translation + QnA demo](https://user-images.githubusercontent.com/87589755/187067522-0fe8ad7c-4b5f-4fd2-8a12-a6191a145931.gif)

