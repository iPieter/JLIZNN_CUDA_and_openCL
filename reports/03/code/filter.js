const amqp = require('amqp');

let connection = amqp.createConnection({ url: 'amqp://bob:bob@10.128.53.171:5672' }, { defaultExchangeName: "amq.topic" });

let on_error = function (e) {
    console.log('error ' + e);
};

// add this for better debuging
connection.on('error', on_error);

// Wait for connection to become established.
connection.on('ready', function () {
    let current = 'text';

    connection.queue('text', function (q) {
        // Catch all messages
        q.bind('amq.topic', 'text');

        connection.exchange('amq.topic', options = {}, function (exchange) {

            // Receive message
            q.subscribe(function (message) {
                fact = JSON.parse(message.data.toString());
                exchange.publish('out', {"value": "Result [" + fact.value + "]"}, {contentType: 'text/plain'});
            });
        });
    });
});
