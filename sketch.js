let pointsX = [];
let pointsY = [];
let a, b;
let optimizer;

function mouseClicked() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, height, 0, 0, 1);
    pointsX.push(x);
    pointsY.push(y);
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
    x = tf.tensor1d(xArray);
    return a.mul(x).add(b);
}

function loss(target, guess) {
    return target.sub(guess).square().mean()
}

function drawLine(a, b) {
    bMapped = map(b.get(), 0, 1, height, 0);
    stroke(255);
    strokeWeight(5);
    line(0, bMapped, width, -a.get() * width + bMapped);
}

function train(optimizer) {
    y = tf.tensor1d(pointsY);
    optimizer.minimize(() => loss(y, predict(pointsX)), true, [a, b]);
}

function setup() {
    createCanvas(700, 700);
    a = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
    optimizer = tf.train.sgd(0.1);

}

function draw() {
    background(10)
    if (pointsX.length > 0) {
        train(optimizer);
        drawPoints();
        drawLine(a, b);
    }
}