function FlattenLayer(inputs) {
    this.inputs = inputs
    this.outputs = inputs[0] * inputs[1] * inputs[2]
}

FlattenLayer.prototype.SetBatchSize = function(batchSize) {
    this.batchSize = batchSize
    this.output = []
    this.dx = []

    for (let i = 0; i < batchSize; i++) {
        this.output[i] = []
        this.dx[i] = InitTensorMemory(this.inputs)

        for (let j = 0; j < this.outputs; j++)
            this.output[i][j] = 0
    }
}

FlattenLayer.prototype.Forward = function(x) {
    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        let index = 0

        for (let i = 0; i < this.inputs[0]; i++)
            for (let j = 0; j < this.inputs[1]; j++)
                for (let k = 0; k < this.inputs[2]; k++)
                    this.output[batchIndex][index++] = x[batchIndex][i][j][k]
    }
}

FlattenLayer.prototype.ForwardOnce = function(x) {
    let output = []
    let index = 0

    for (let i = 0; i < this.inputs[0]; i++)
        for (let j = 0; j < this.inputs[1]; j++)
            for (let k = 0; k < this.inputs[2]; k++)
                output[index++] = x[i][j][k]

    return output
}

FlattenLayer.prototype.Backward = function(dout, x, calc_dX) {
    if (!calc_dX)
        return

    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        let index = 0

        for (let i = 0; i < this.inputs[0]; i++)
            for (let j = 0; j < this.inputs[1]; j++)
                for (let k = 0; k < this.inputs[2]; k++)
                    this.dx[batchIndex][i][j][k] = dout[batchIndex][index++]
    }
}


FlattenLayer.prototype.ZeroGradients = function() {
    
}

FlattenLayer.prototype.UpdateWeights = function(optimizer) {
    
}

FlattenLayer.prototype.SetWeights = function(layer) {
    
}
