function MaxPoolingLayer(inputs, scale) {
    this.inputs = inputs
    this.outputs = [
        Math.floor((inputs[0] + scale - 1) / scale),
        Math.floor((inputs[1] + scale - 1) / scale),
        inputs[2]
    ]

    this.scale = scale
    this.di = []
    this.dj = []

    for (let i = 0; i < inputs[0]; i++)
        this.di[i] = Math.floor(i / this.scale)

    for (let i = 0; i < inputs[1]; i++)
        this.dj[i] = Math.floor(i / this.scale)
}

MaxPoolingLayer.prototype.SetBatchSize = function(batchSize) {
    this.batchSize = batchSize
    this.output = []
    this.dx = []

    for (let i = 0; i < batchSize; i++) {
        this.output[i] = InitTensorMemory(this.outputs)
        this.dx[i] = InitTensorMemory(this.inputs)
    }
}

MaxPoolingLayer.prototype.Forward = function(x) {
    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        for (let i = 0; i < this.inputs[0]; i += this.scale) {
            for (let j = 0; j < this.inputs[1]; j += this.scale) {
                for (let k = 0; k < this.inputs[2]; k++) {
                    let imax = i
                    let jmax = j
                    let max = x[batchIndex][i][j][k]

                    for (let ii = i; ii < i + this.scale && ii < this.inputs[0]; ii++) {
                        for (let jj = j; jj < j + this.scale && jj < this.inputs[1]; jj++) {
                            let value = x[batchIndex][ii][jj][k]
                            this.dx[batchIndex][ii][jj][k] = 0

                            if (value > max) {
                                max = value
                                imax = ii
                                jmax = jj
                            }
                        }
                    }

                    this.output[batchIndex][this.di[i]][this.dj[j]][k] = max
                    this.dx[batchIndex][imax][jmax][k] = 1
                }
            }
        }
    }
}

MaxPoolingLayer.prototype.ForwardOnce = function(x) {
    let output = InitTensorMemory(this.outputs)

    for (let i = 0; i < this.inputs[0]; i += this.scale) {
        for (let j = 0; j < this.inputs[1]; j += this.scale) {
            for (let k = 0; k < this.inputs[2]; k++) {
                let imax = i
                let jmax = j
                let max = x[i][j][k]

                for (let ii = i; ii < i + this.scale && ii < this.inputs[0]; ii++) {
                    for (let jj = j; jj < j + this.scale && jj < this.inputs[1]; jj++) {
                        let value = x[ii][jj][k]

                        if (value > max) {
                            max = value
                            imax = ii
                            jmax = jj
                        }
                    }
                }

                output[this.di[i]][this.dj[j]][k] = max
            }
        }
    }

    return output
}

MaxPoolingLayer.prototype.Backward = function(dout, x, calc_dX) {
    if (!calc_dX)
        return

    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++)
        for (let i = 0; i < this.inputs[0]; i++)
            for (let j = 0; j < this.inputs[1]; j++)
                for (let k = 0; k < this.inputs[2]; k++)
                    this.dx[batchIndex][i][j][k] *= dout[batchIndex][this.di[i]][this.dj[j]][k]
}

MaxPoolingLayer.prototype.ZeroGradients = function() {
    
}

MaxPoolingLayer.prototype.UpdateWeights = function(optimizer) {
    
}

MaxPoolingLayer.prototype.SetWeights = function(layer) {
    
}
