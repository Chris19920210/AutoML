import pika
import json
from bayesian_research.utils import MyEncoder


class JobMq:

    def __init__(self):
        self.job_queue = "hyperjob"
        self.back_queue = "feedback"
        host = "172.21.11.1"
        port = 5672
        user = "root"
        passwd = "root"
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=host,
            port=port,
            user=user,
            password=passwd
        ))
        self.channel = self.connection.channel()

    def init_job(self, msg):
        self.publish_job(msg)
        print('init_job: %s' % msg)

    def consume(self, func):
        self.channel.basic_consume(func,
                                   queue=self.back_queue,
                                   no_ack=False)
        self.channel.start_consuming()
        print('start consume: %s' % func)

    def publish_job(self, job):
        self.channel.basic_publish(exchange='',
                                   routing_key=self.job_queue,
                                   body=json.dumps(job, cls=MyEncoder))

        print('publish_job: %s' % job)

    def close(self):
        self.connection.close()
