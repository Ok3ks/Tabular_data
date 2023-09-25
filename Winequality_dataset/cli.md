Incomplete:

use this to learn how to setup github teaching repository.

Using AWS and MLFlow for MLOps
Using Prefect and Streaming Application on MLOps

```bash 

KINESIS_TEST="winequality"

aws kinesis put-record \
    --stream-name ${KINESIS_TEST} \
    --partition-key 1 \
    --data '{
        'wine' : {
    "sulphates": 4,
    "chlorides": 5,
    "volatile acidity": 6,
    "quality": 7,
    "alcohol": 8,
    "pH": 9, 
    "residual sugar": 10,
    "total sulfur dioxide":12
        }
        'wine_id': 22
}
{
"ShardId": "shardId-000000000000",
"SequenceNumber": "49639661238418506276826155738646746753646419808385761282"
}