FROM public.ecr.aws/lambda/python:3.9

COPY weights ${LAMBDA_TASK_ROOT}/weights

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY lambda.py ${LAMBDA_TASK_ROOT}

COPY anonymizer ${LAMBDA_TASK_ROOT}/anonymizer

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda.handler" ]
