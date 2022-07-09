function SoftmaxLayer(size) {
    this.size = size
    this.outputs = size
}

SoftmaxLayer.prototype.SetBatchSize = function(batchSize) {
    this.batchSize = batchSize

    this.output = []
    this.dx = []

    for (let i = 0; i < batchSize; i++) {
        this.output[i] = []
        this.dx[i] = []

        for (let j = 0; j < this.size; j++) {
            this.output[i][j] = 0
            this.dx[i][j] = 0
        }
    }
}

SoftmaxLayer.prototype.Forward = function(x) {
    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        let sum = 0

        for (let i = 0; i < this.size; i++) {
            this.output[batchIndex][i] = Math.exp(x[batchIndex][i])
            sum += this.output[batchIndex][i]
        }

        for (let i = 0; i < this.size; i++)
            this.output[batchIndex][i] /= sum
    }
}

SoftmaxLayer.prototype.ForwardOnce = function(x) {
    let output = []
    let sum = 0

    for (let i = 0; i < this.size; i++) {
        output[i] = Math.exp(x[i])
        sum += output[i]
    }

    for (let i = 0; i < this.size; i++)
        output[i] /= sum

    return output
}

SoftmaxLayer.prototype.Backward = function(dout, x, calc_dX) {
    if (!calc_dX)
        return

    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        for (let i = 0; i < this.size; i++) {
            let sum = 0

            for (let j = 0; j < this.size; j++)
                sum += dout[batchIndex][j] * this.output[batchIndex][i] * ((i == j) - this.output[batchIndex][j]);

            this.dx[batchIndex][i] = sum
        }
    }
}

SoftmaxLayer.prototype.ZeroGradients = function() {
    
}

SoftmaxLayer.prototype.UpdateWeights = function(optimizer) {
    
}

SoftmaxLayer.prototype.SetWeights = function(layer) {
    
}
