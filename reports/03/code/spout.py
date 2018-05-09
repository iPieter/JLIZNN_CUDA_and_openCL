#!/usr/bin/env python

import pika
import time
from random import randint
import json

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()

for i in range(0,500):
	target = randint(10, 10000000)
	print("Publishing: {}".format(target))
	channel.basic_publish(exchange='amq.topic',routing_key='transform',body=json.dumps({'value' : target}))
	time.sleep(0.05)
connection.close()
