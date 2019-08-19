########################### DEPENDENCIES  ###########################

# initiates the AWS command line commands (via terminal)
aws configure

# importing the Amazon S3 service
s3 = boto3.resource('s3')