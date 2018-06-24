let pointsX = [];
let pointsY = [];
let polynomialDegree = 2;
let coefficients = new Array(polynomialDegree);
let optimizer;
let xsToDraw = [];
polynomialDegree += 1;

function setupPointsToDraw() {
    for (let i = 0.01; i < 1; i += 0.01) {
        xsToDraw.push(i);
    }
}

function addPoints() {
    // pointsX.push(0.1);
    // pointsY.push(0.2);

    // pointsX.push(0.3);
    // pointsY.push(0.5);

    // pointsX.push(0.5);
    // pointsY.push(0.2);

    // pointsX.push(0.7);
    // pointsY.push(0.6);

    // pointsX.push(0.9);
    // pointsY.push(0.1);
}

function mouseClicked() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, height, 0, 0, 1);
    pointsX.push(x);
    pointsY.push(y);
}

function setup() {
    createCanvas(700, 700);
    // frameRate(1);
    for (let i = 0; i < polynomialDegree; i++) {
        coefficients[i] = tf.variable(tf.scalar(random(1)));
    }
    optimizer = tf.train.adam(0.02);
    addPoints();
    setupPointsToDraw();

}

function train(optimizer) {
    let y = tf.tensor1d(pointsY);
    optimizer.minimize(() => loss(y, predict(pointsX)), true);
    y.dispose();
}

function loss(target, guess) {
    return target.sub(guess).square().mean()
}

function predict(xArray) {
    let x = tf.tensor1d(xArray);
    let result = tf.fill([xArray.length], 0);
    for (let i = 0; i < polynomialDegree; i++) {
        let degree = tf.scalar(i);
        result = result.add(x.pow(degree).mul(coefficients[i]));
        // console.log(i);
        // console.log("coeffitients");
        // coefficients[i].print();
        // console.log("degree");
        // degree.print();
        // console.log("x");
        // x.print();
        // console.log("");
        // console.log("result:")
        // result.print();
        // console.log("-----------------");
        degree.dispose();
    }
    x.dispose();
    // console.log("/////////////////////////");
    return result;
}




function drawPoints() {
    fill(255);
    for (let i = 0; i < pointsX.length; i++) {
        let x = map(pointsX[i], 0, 1, 0, width);
        let y = map(pointsY[i], 0, 1, height, 0);
        ellipse(x, y, 10, 10);
    }
}

function drawLine() {

    noFill();
    strokeWeight(2);
    stroke(255);
    beginShape();
    predict(xsToDraw).data().then(ysToDraw => {
        for (let i = 0; i < 99; i += 1) {
            let x = map(xsToDraw[i], 0, 1, 0, width);
            let y = map(ysToDraw[i], 0, 1, height, 0);
            vertex(x, y);
        }
        endShape();
    });


}



function draw() {
    tf.tidy(() => {
        background(10)
        if (pointsX.length > 0) {
            train(optimizer);
            drawPoints();
            drawLine();
        }
    });
}