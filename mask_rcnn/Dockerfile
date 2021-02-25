#Set origin 
FROM python:3

#set work dir
WORKDIR /mask_rcnn

#copy the current directory to image
COPY . /mask_rcnn

#install all the requirements
RUN pip3 install -r requirements.txt

#export to port 8000
EXPOSE 8000

#run train_model
CMD ["python", "train_model.py"]
