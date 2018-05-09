#!/usr/bin/env python

# -*- coding: utf-8 -*-

import pika

import datetime

import time

from sys import exit



now = datetime.datetime.now()



credentials = pika.PlainCredentials('bob', 'bob')

connection = pika.BlockingConnection(pika.ConnectionParameters(

        'localhost', '5672', '/', credentials))

channel = connection.channel()



#channel.basic_qos(prefetch_count=500)



# Re-declare the queue with passive flag

res = channel.queue_declare(

        queue="out",

        durable=True,

        exclusive=False,

        auto_delete=False,

        passive=True

    )



messages = res.method.message_count



def callback(ch, method, properties, body):

    with open( "measurements_rabbitmq.csv", "a") as myfile:

    	myfile.write( "{};{}\n".format( time.time(),  body))

        #global count

        #count = count + 1

        #print_progress(count, messages)	



channel.basic_consume(callback,

                      queue='out',

                      no_ack=True)



channel.start_consuming()
