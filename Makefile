install:
	pip3 install --upgrade pip &&\
		pip3 install -r requirements.txt

lint:
	pylint --disable=R,C,W1203,W1202 **.py

all: install lint