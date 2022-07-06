function FullyConnectedLayer(inputs, outputs, activation) {
    this.inputs = inputs
    this.outputs = outputs
    this.activation = activation

    this.w = [] // весовые коэффициенты
    this.b = [] // веса смещения

    this.InitWeights()
}

FullyConnectedLayer.prototype.InitWeights = function() {
    let lim = Math.sqrt(6 / this.inputs)

    for (let i = 0; i < this.outputs; i++) {
        this.b[i] = new Weight(-lim, lim)
        this.w[i] = []

        for (let j = 0; j < this.inputs; j++)
            this.w[i][j] = new Weight(-lim, lim)
    }
}

FullyConnectedLayer.prototype.SetBatchSize = function(batchSize) {
    this.batchSize = batchSize

    this.output = []
    this.df = []
    this.dx = []

    for (let i = 0; i < batchSize; i++) {
        this.output[i] = []
        this.df[i] = []
        this.dx[i] = []

        for (let j = 0; j < this.outputs; j++) {
            this.output[i][j] = 0
            this.df[i][j] = 0
        }

        for (let j = 0; j < this.inputs; j++) {
            this.dx[i][j] = 0
        }
    }
}

FullyConnectedLayer.prototype.Activate = function(batchIndex, i, value) {
    if (this.activation == '') {
        this.output[batchIndex][i] = value
        this.df[batchIndex][i] = 1
    }
    else if (this.activation == 'sigmoid') {
        value = 1 / (1 + Math.exp(-value))
        this.output[batchIndex][i] = value
        this.df[batchIndex][i] = value * (1 - value)
    }
    else if (this.activation == 'tanh') {
        value = Math.tanh(value)
        this.output[batchIndex][i] = value
        this.df[batchIndex][i] = 1 - value * value
    }
    else if (this.activation == 'relu') {
        if (value > 0) {
            this.output[batchIndex][i] = value
            this.df[batchIndex][i] = 1
        }
        else {
            this.output[batchIndex][i] = 0
            this.df[batchIndex][i] = 0
        }
    }
    else if (this.activation == 'leaky-relu') {
        if (value > 0) {
            this.output[batchIndex][i] = value
            this.df[batchIndex][i] = 1
        }
        else {
            this.output[batchIndex][i] = 0.01 * value
            this.df[batchIndex][i] = 0.01
        }
    }
    else if (this.activation == 'elu') {
        if (value > 0) {
            this.output[batchIndex][i] = value
            this.df[batchIndex][i] = 1
        }
        else {
            this.output[batchIndex][i] = Math.exp(value) - 1
            this.df[batchIndex][i] = Math.exp(value)
        }
    }
    else if (this.activation == 'swish') {
        let sigmoid = 1.0 / (1 + Math.exp(-value))

        this.output[batchIndex][i] = value * sigmoid
        this.df[batchIndex][i] = sigmoid + value * sigmoid * (1 - sigmoid)
    }
    else if (this.activation == 'softplus') {
        this.output[batchIndex][i] = Math.log(1 + Math.exp(value))
        this.df[batchIndex][i] = 1.0 / (1 + Math.exp(-value))
    }
    else if (this.activation == 'softsign') {
        this.output[batchIndex][i] = value / (1 + Math.abs(value))
        this.df[batchIndex][i] = 1.0 / Math.pow(1 + Math.abs(value), 2)
    }
    else if (this.activation == 'abs') {
        this.output[batchIndex][i] = Math.abs(value)
        this.df[batchIndex][i] = Math.sign(value)
    }
}

FullyConnectedLayer.prototype.ActivateOnce = function(value) {
    if (this.activation == 'sigmoid')
        return 1 / (1 + Math.exp(-value))

    if (this.activation == 'tanh')
        return Math.tanh(value)

    if (this.activation == 'relu')
        return Math.max(0, value)

    if (this.activation == 'leaky-relu')
        return value > 0 ? value : 0.01 * value

    if (this.activation == 'elu')
        return value > 0 ? value : Math.exp(value) - 1

    if (this.activation == 'swish')
        return value / (1 + Math.exp(-value))

    if (this.activation == 'softplus')
        return Math.log(1 + Math.exp(value))

    if (this.activation == 'softsign')
        return value / (1 + Math.abs(value))

    if (this.activation == 'abs')
        return Math.abs(value)

    return value
}

FullyConnectedLayer.prototype.Forward = function(x) {
    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        for (let i = 0; i < this.outputs; i++) {
            let sum = this.b[i].value

            for (let j = 0; j < this.inputs; j++)
                sum += this.w[i][j].value * x[batchIndex][j]

            this.Activate(batchIndex, i, sum)
        }
    }
}

FullyConnectedLayer.prototype.ForwardOnce = function(x) {
    let output = []

    for (let i = 0; i < this.outputs; i++) {
        let sum = this.b[i].value

        for (let j = 0; j < this.inputs; j++)
            sum += this.w[i][j].value * x[j]

        output[i] = this.ActivateOnce(sum)
    }

    return output
}

FullyConnectedLayer.prototype.Backward = function(dout, x, calc_dX) {
    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        for (let i = 0; i < this.outputs; i++) {
            let delta = dout[batchIndex][i] * this.df[batchIndex][i]

            for (let j = 0; j < this.inputs; j++)
                this.w[i][j].grad += delta * x[batchIndex][j]

            this.b[i].grad += delta
        }
    }

    if (!calc_dX)
        return

    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        for (let j = 0; j < this.inputs; j++) {
            let sum = 0

            for (let i = 0; i < this.outputs; i++)
                sum += this.w[i][j].value * dout[batchIndex][i] * this.df[batchIndex][i]

            this.dx[batchIndex][j] = sum
        }
    }
}

FullyConnectedLayer.prototype.ZeroGradients = function() {
    for (let i = 0; i < this.outputs; i++) {
        for (let j = 0; j < this.inputs; j++)
            this.w[i][j].ZeroGrad()

        this.b[i].ZeroGrad()
    }
}

FullyConnectedLayer.prototype.UpdateWeights = function(optimizer) {
    for (let i = 0; i < this.outputs; i++) {
        for (let j = 0; j < this.inputs; j++)
            optimizer.Update(this.w[i][j], this.batchSize)

        optimizer.Update(this.b[i], this.batchSize)
    }
}

FullyConnectedLayer.prototype.SetWeights = function(layer) {
    for (let i = 0; i < this.outputs; i++) {
        for (let j = 0; j < this.inputs; j++)
            this.w[i][j] = layer.w[i][j].Copy()

        this.b[i] = layer.b[i].Copy()
    }
}
