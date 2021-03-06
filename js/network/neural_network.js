function NeuralNetwork(inputs) {
    this.inputs = inputs
    this.outputs = inputs
    this.layers = []
}

NeuralNetwork.prototype.AddLayer = function(description) {
    let layer = null

    if (description.type == 'fc') {
        layer = new FullyConnectedLayer(this.outputs, description.size, description.activation)
    }
    else if (description.type == 'conv') {
        layer = new ConvLayer(this.outputs, description.fc, description.fs, description.padding, description.stride)
    }
    else if (description.type == 'maxpool') {
        layer = new MaxPoolingLayer(this.outputs, description.scale)
    }
    else if (description.type == 'softmax') {
        layer = new SoftmaxLayer(this.outputs)
    }
    else if (description.type == 'flatten') {
        layer = new FlattenLayer(this.outputs)
    }
    else 
        throw new Error(`Unknown layer type "${description.type}"`)

    this.outputs = layer.outputs
    this.layers.push(layer)
}

NeuralNetwork.prototype.SetBatchSize = function(batchSize) {
    this.batchSize = batchSize

    for (let layer of this.layers) {
        layer.SetBatchSize(batchSize)
    }
}

NeuralNetwork.prototype.ZeroGradients = function() {
    for (let layer of this.layers) {
        layer.ZeroGradients()
    }
}

NeuralNetwork.prototype.SetWeights = function(network) {
    for (let i = 0; i < this.layers.length; i++)
        this.layers[i].SetWeights(network.layers[i])
}

NeuralNetwork.prototype.Forward = function(x) {
    this.layers[0].Forward(x)

    for (let i = 1; i < this.layers.length; i++)
        this.layers[i].Forward(this.layers[i - 1].output)

    return this.layers[this.layers.length - 1].output
}

NeuralNetwork.prototype.Backward = function(x, deltas) {
    let last = this.layers.length - 1

    if (last == 0) {
        this.layers[last].Backward(deltas, x, true)
    }
    else {
        this.layers[last].Backward(deltas, this.layers[last - 1].output, true)

        for (let i = last - 1; i > 0; i--)
            this.layers[i].Backward(this.layers[i + 1].dx, this.layers[i - 1].output, true)

        this.layers[0].Backward(this.layers[1].dx, x, false)
    }
}

NeuralNetwork.prototype.UpdateWeights = function(optimizer) {
    for (let layer of this.layers)
        layer.UpdateWeights(optimizer)
}

NeuralNetwork.prototype.Predict = function(x) {
    x = this.layers[0].ForwardOnce(x)

    for (let i = 1; i < this.layers.length; i++)
        x = this.layers[i].ForwardOnce(x)

    return x
}

NeuralNetwork.prototype.CalculateLoss = function(y, t, deltas, L) {
    let loss = 0

    for (let i = 0; i < y.length; i++) {
        deltas[i] = []
        loss += L.EvaluateDeltas(y[i], t[i], deltas[i])
    }

    return loss
}

NeuralNetwork.prototype.TrainOnBatch = function(x, y, optimizer, L) {
    return this.TrainOnBatchWithOutput(x, y, this.Forward(x), optimizer, L)
}

NeuralNetwork.prototype.TrainOnBatchWithOutput = function(x, y, output, optimizer, L) {
    let deltas = []
    let loss = this.CalculateLoss(output, y, deltas, L)

    this.ZeroGradients()
    this.Backward(x, deltas)
    this.UpdateWeights(optimizer)

    return loss
}
