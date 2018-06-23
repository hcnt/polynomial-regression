let pointsX = [];
let pointsY = [];
let a, b, c;
let optimizer;

function mouseClicked() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, height, 0, 0, 1);
    pointsX.push(x);
    pointsY.push(y);
}

function setup() {
    createCanvas(700, 700);
    a = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
    c = tf.variable(tf.scalar(random(1)));
    optimizer = tf.train.adam(0.05);

}

function drawPoints() {
    fill(255);
    for (let i = 0; i < pointsX.length; i++) {
        let x = map(pointsX[i], 0, 1, 0, width);
        let y = map(pointsY[i], 0, 1, height, 0);
        ellipse(x, y, 10, 10);
    }
}

function predict(xArray) {
    let x = tf.tensor1d(xArray);
    return a.mul(x.square()).add(b.mul(x)).add(c);
}

function loss(target, guess) {
    return target.sub(guess).square().mean()
}

function drawLine(aScalar, bScalar, cScalar) {

    let c = cScalar.get()
    let b = bScalar.get();
    let a = aScalar.get();

    noFill();
    strokeWeight(2);
    stroke(255);
    beginShape();
    for (let i = 0; i < 1; i += 0.01) {
        let x = map(i, 0, 1, 0, width);
        let y = map(a * (i ** 2) + b * i + c, 0, 1, height, 0);
        vertex(x, y);
    }
    endShape();
}

function train(optimizer) {
    let y = tf.tensor1d(pointsY);
    optimizer.minimize(() => loss(y, predict(pointsX)), true, [a, c, b]);
}


function draw() {
    tf.tidy(() => {
        background(10)
        if (pointsX.length > 0) {
            train(optimizer);
            drawPoints();
            drawLine(a, b, c);
        }
    });
}