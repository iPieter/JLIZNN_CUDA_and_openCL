const amqp = require('amqp');

let connection = amqp.createConnection({ url: 'amqp://bob:bob@10.128.53.171:5672' }, { defaultExchangeName: "amq.topic" });

let on_error = function (e) {
    console.log('error ' + e);
};

let findPrimeFactors = (num) => {
    let arr = [];

    for (var i = 2; i < num; i++) {
        let isPrime
        if (num % i === 0) {
            isPrime = true;
            for (var j = 2; j <= i; j++) {
                if (i % j === 0) {
                    isPrime == false;
                }
            }
        } if (isPrime == true) { arr.push(i) }

    }
    return arr;
}

// add this for better debuging
connection.on('error', on_error);

// Wait for connection to become established.
connection.on('ready', function () {
    let current = 'transform';

    connection.queue(current, function (q) {
        // Catch all messages
        q.bind('amq.topic', current);

        connection.exchange('amq.topic', options = {}, function (exchange) {

            // Receive message
            q.subscribe(function (message) {
                fact = JSON.parse(message.data.toString());
                exchange.publish('filter', {"value": findPrimeFactors(fact.value)}, {contentType: 'text/plain'});
            });
        });
    });
});
