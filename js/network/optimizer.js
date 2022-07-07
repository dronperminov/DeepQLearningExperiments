function Optimizer(learningRate, lambda = 0, algorithm = 'sgd', regularizationType = 'l2') {
    this.learningRate = learningRate
    this.lambda = lambda
    this.algorithm = algorithm
    this.regularizationType = regularizationType
    this.update = this.UpdateSGD
    this.epoch = 1

    if (this.algorithm == 'sgdm') {
        this.beta = 0.9
        this.update = this.UpdateMomentumSGD
    }
    else if (this.algorithm == 'nag') {
        this.beta = 0.9
        this.update = this.UpdateNAG
    }
    else if (this.algorithm == 'adam') {
        this.beta1 = 0.9
        this.beta2 = 0.999
        this.update = this.UpdateAdam
    }
    else if (this.algorithm == 'nadam') {
        this.beta1 = 0.9
        this.beta2 = 0.999
        this.update = this.UpdateNAdam
    }
    else if (this.algorithm == 'adagrad') {
        this.update = this.UpdateAdagrad
    }
    else if (this.algorithm == 'adamax') {
        this.beta1 = 0.9
        this.beta2 = 0.999
        this.update = this.UpdateAdaMax
    }
    else if (this.algorithm == 'rmsprop') {
        this.beta1 = 0.9
        this.update = this.UpdateRMSprop
    }
    else if (this.algorithm == 'amsgrad') {
        this.beta1 = 0.9
        this.beta2 = 0.999
        this.update = this.UpdateAMSgrad
    }
}

Optimizer.prototype.SetLearningRate = function(learningRate) {
    this.learningRate = learningRate
}

Optimizer.prototype.SetRegularization = function(lambda) {
    this.lambda = lambda
}

Optimizer.prototype.SetRegularizationType = function(regularizationType) {
    this.regularizationType = regularizationType
}

Optimizer.prototype.UpdateSGD = function(weight) {
    weight.value -= this.learningRate * weight.grad
}

Optimizer.prototype.UpdateMomentumSGD = function(weight) {
    weight.param1 = this.beta * weight.param1 + this.learningRate * weight.grad
    weight.value -= weight.param1
}

Optimizer.prototype.UpdateNAG = function(weight) {
    let prev = weight.param1
    weight.param1 = this.beta * weight.param1 - this.learningRate * weight.grad
    weight.value += this.beta * (weight.param1 - prev) + weight.param1
}

Optimizer.prototype.UpdateNAdam = function(weight) {
    let mt1 = weight.param1 / (1 - Math.pow(this.beta1, this.epoch))

    weight.param1 = this.beta1 * weight.param1 + (1 - this.beta1) * weight.grad
    weight.param2 = this.beta2 * weight.param2 + (1 - this.beta2) * weight.grad * weight.grad

    let Vt = weight.param1 / (1 - Math.pow(this.beta1, this.epoch))
    let St = weight.param2 / (1 - Math.pow(this.beta2, this.epoch))

    weight.value -= this.learningRate * (this.beta1 * mt1 + (1 - this.beta1) / (1 - Math.pow(this.beta1, this.epoch)) * weight.grad) / (Math.sqrt(St) + 1e-7)
}

Optimizer.prototype.UpdateAdam = function(weight) {
    weight.param1 = this.beta1 * weight.param1 + (1 - this.beta1) * weight.grad
    weight.param2 = this.beta2 * weight.param2 + (1 - this.beta2) * weight.grad * weight.grad

    let mt = weight.param1 / (1 - Math.pow(this.beta1, this.epoch))
    let vt = weight.param2 / (1 - Math.pow(this.beta2, this.epoch))

    weight.value -= this.learningRate * mt / (Math.sqrt(vt) + 1e-8)
}

Optimizer.prototype.UpdateAdagrad = function(weight) {
    weight.param1 += weight.grad * weight.grad
    weight.value -= this.learningRate * weight.grad / Math.sqrt(weight.param1 + 1e-8)
}

Optimizer.prototype.UpdateAdaMax = function(weight) {
    weight.param1 = this.beta1 * weight.param1 + (1 - this.beta1) * weight.grad
    weight.param2 = Math.max(this.beta2 * weight.param2, Math.abs(weight.grad))

    let mt = weight.param1 / (1 - Math.pow(this.beta1, this.epoch))

    weight.value -= this.learningRate * mt / (weight.param2 + 1e-8)
}

Optimizer.prototype.UpdateRMSprop = function(weight) {
    weight.param1 = weight.param1 * this.beta1 + (1 - this.beta1) * weight.grad * weight.grad
    weight.value -= this.learningRate * weight.grad / Math.sqrt(weight.param1 + 1e-8)
}

Optimizer.prototype.UpdateAMSgrad = function(weight) {
    weight.param1 = this.beta1 * weight.param1 + (1 - this.beta1) * weight.grad
    weight.param2 = this.beta2 * weight.param2 + (1 - this.beta2) * weight.grad * weight.grad
    weight.param3 = Math.max(weight.param2, weight.param3)

    weight.value -= this.learningRate * weight.param1 / (Math.sqrt(weight.param3) + 1e-8)
}

Optimizer.prototype.Update = function(weight, batchSize) {
    weight.grad /= batchSize

    if (this.regularizationType == 'l1') {
        weight.grad += this.lambda * Math.sign(weight.value)
    }
    else if (this.regularizationType == 'l2') {
        weight.grad += this.lambda * weight.value
    }

    this.update(weight)
}

Optimizer.prototype.UpdateEpoch = function() {
    this.epoch++
}