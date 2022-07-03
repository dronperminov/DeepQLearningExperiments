function MSE() {
    console.log("Use MSE loss")
}

function MAE() {
    console.log("Use MAE loss")
}

function LogCosh() {
    console.log("Use logcosh loss")
}

function Huber(delta) {
    this.delta = delta
    console.log(`Use Huber loss with delta=${delta}`)
}

MSE.prototype.EvaluateDeltas = function(y, t, deltas) {
    let loss = 0

    for (let i = 0; i < y.length; i++) {
        let e = y[i] - t[i]
        deltas[i] = e
        loss += 0.5*e*e
    }

    return loss
}

MAE.prototype.EvaluateDeltas = function(y, t, deltas) {
    let loss = 0

    for (let i = 0; i < y.length; i++) {
        let e = y[i] - t[i]
        deltas[i] = Math.sign(e)
        loss += Math.abs(e)
    }

    return loss
}

LogCosh.prototype.EvaluateDeltas = function(y, t, deltas) {
    let loss = 0

    for (let i = 0; i < y.length; i++) {
        let e = y[i] - t[i]
        deltas[i] = Math.tanh(e)
        loss += Math.log(Math.cosh(e))
    }

    return loss
}

Huber.prototype.EvaluateDeltas = function(y, t, deltas) {
    let loss = 0

    for (let i = 0; i < y.length; i++) {
        let e = y[i] - t[i]

        if (Math.abs(e) < this.delta) {
            deltas[i] = e
            loss += 0.5*e*e
        }
        else {
            deltas[i] = this.delta * Math.sign(e)
            loss += this.delta * (Math.abs(e) - 0.5 * this.delta)
        }
    }

    return loss
}
