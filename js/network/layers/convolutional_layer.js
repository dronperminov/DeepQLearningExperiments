function ConvLayer(inputs, fc, fs, padding, stride) {
    if (!(inputs instanceof Array) || inputs.length != 3)
        throw new Error("Invalid conv layer inputs")

    this.inputs = inputs
    this.outputs = [
        Math.floor((inputs[0] - fs + 2 * padding) / stride) + 1, 
        Math.floor((inputs[1] - fs + 2 * padding) / stride) + 1,
        fc
    ]

    this.deltasSize = [
        stride * (this.outputs[0] - 1) + 1,
        stride * (this.outputs[1] - 1) + 1,
        this.outputs[2]
    ]

    this.fc = fc
    this.fs = fs
    this.fd = inputs[2] // каналы последние

    this.padding = padding
    this.stride = stride

    this.w = []
    this.b = []

    this.InitWeights()
}

ConvLayer.prototype.InitWeights = function() {
    let lim = Math.sqrt(6 / (this.fs * this.fs * this.fd))

    for (let index = 0; index < this.fc; index++) {
        this.w[index] = []

        for (let i = 0; i < this.fs; i++) {
            this.w[index][i] = []

            for (let j = 0; j < this.fs; j++) {
                this.w[index][i][j] = []

                for (let k = 0; k < this.fd; k++)
                    this.w[index][i][j][k] = new Weight(-lim, lim)
            }
        }

        this.b.push(new Weight(-lim, lim))
    }
}

ConvLayer.prototype.SetBatchSize = function(batchSize) {
    this.batchSize = batchSize

    this.output = []
    this.df = []
    this.dx = []

    this.deltas = []

    for (let index = 0; index < batchSize; index++) {
        this.output[index] = InitTensorMemory(this.outputs)
        this.deltas[index] = InitTensorMemory(this.deltasSize)
        this.df[index] = InitTensorMemory(this.outputs)
        this.dx[index] = InitTensorMemory(this.inputs)
    }
}

ConvLayer.prototype.Forward = function(x) {
    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        for (let i = 0; i < this.outputs[0]; i++) {
            for (let j = 0; j < this.outputs[1]; j++) {
                for (let f = 0; f < this.fc; f++) {
                    let sum = this.b[f].value

                    for (let k = 0; k < this.fs; k++) {
                        let i0 = this.stride * i + k - this.padding

                        if (i0 < 0 || i0 >= this.inputs[0])
                            continue

                        for (let l = 0; l < this.fs; l++) {
                            let j0 = this.stride * j + l - this.padding

                            if (j0 < 0 || j0 >= this.inputs[1])
                                continue

                            for (let c = 0; c < this.fd; c++)
                                sum += x[batchIndex][i0][j0][c] * this.w[f][k][l][c].value
                        }
                    }

                    if (sum > 0) {
                        this.output[batchIndex][i][j][f] = sum
                        this.df[batchIndex][i][j][f] = 1
                    }
                    else {
                        this.output[batchIndex][i][j][f] = 0
                        this.df[batchIndex][i][j][f] = 0
                    }
                }
            }
        }
    }
}

ConvLayer.prototype.ForwardOnce = function(x) {
    let output = InitTensorMemory(this.outputs)

    for (let i = 0; i < this.outputs[0]; i++) {
        for (let j = 0; j < this.outputs[1]; j++) {
            for (let f = 0; f < this.fc; f++) {
                let sum = this.b[f].value

                for (let k = 0; k < this.fs; k++) {
                    let i0 = this.stride * i + k - this.padding

                    if (i0 < 0 || i0 >= this.inputs[0])
                        continue

                    for (let l = 0; l < this.fs; l++) {
                        let j0 = this.stride * j + l - this.padding

                        if (j0 < 0 || j0 >= this.inputs[1])
                            continue

                        for (let c = 0; c < this.fd; c++) {
                            sum += x[i0][j0][c] * this.w[f][k][l][c].value
                        }
                    }
                }

                output[i][j][f] = Math.max(sum, 0)
            }
        }
    }

    return output
}

ConvLayer.prototype.Backward = function(dout, x, calc_dX) {
    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++)
        for (let i = 0; i < this.outputs[0]; i++)
            for (let j = 0; j < this.outputs[1]; j++)
                for (let d = 0; d < this.outputs[2]; d++)
                    this.deltas[batchIndex][i * this.stride][j * this.stride][d] = dout[batchIndex][i][j][d] * this.df[batchIndex][i][j][d]

    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        for (let k = 0; k < this.deltasSize[0]; k++) {
            for (let l = 0; l < this.deltasSize[1]; l++) {
                for (let f = 0; f < this.fc; f++) {
                    let delta = this.deltas[batchIndex][k][l][f]

                    for (let i = 0; i < this.fs; i++) {
                        let i0 = i + k - this.padding

                        if (i0 < 0 || i0 >= this.inputs[0])
                            continue

                        for (let j = 0; j < this.fs; j++) {
                            let j0 = j + l - this.padding

                            if (j0 < 0 || j0 >= this.inputs[1])
                                continue

                            for (let c = 0; c < this.fd; c++)
                                this.w[f][i][j][c].grad += delta * x[batchIndex][i0][j0][c]
                        }
                    }

                    this.b[f].grad += delta
                }
            }
        }
    }

    if (!calc_dX)
        return

    let pad = this.fs - 1 - this.padding

    for (let batchIndex = 0; batchIndex < this.batchSize; batchIndex++) {
        for (let i = 0; i < this.inputs[0]; i++) {
            for (let j = 0; j < this.inputs[1]; j++) {
                for (let c = 0; c < this.fd; c++) {
                    let sum = 0

                    for (let k = 0; k < this.fs; k++) {
                        let i0 = i + k - pad

                        if (i0 < 0 || i0 >= this.deltasSize[0])
                            continue

                        for (let l = 0; l < this.fs; l++) {
                            let j0 = j + l - pad

                            if (j0 < 0 || j0 >= this.deltasSize[1])
                                continue

                            for (let f = 0; f < this.fc; f++)
                                sum += this.w[f][this.fs - 1 - k][this.fs - 1 - l][c].value * this.deltas[batchIndex][i0][j0][f]
                        }
                    }

                    this.dx[batchIndex][i][j][c] = sum
                }
            }
        }
    }
}

ConvLayer.prototype.ZeroGradients = function() {
    for (let index = 0; index < this.fc; index++) {
        for (let i = 0; i < this.fs; i++)
            for (let j = 0; j < this.fs; j++)
                for (let k = 0; k < this.fd; k++)
                    this.w[index][i][j][k].ZeroGrad()

        this.b[index].ZeroGrad()
    }
}

ConvLayer.prototype.UpdateWeights = function(optimizer) {
    for (let index = 0; index < this.fc; index++) {
        for (let i = 0; i < this.fs; i++)
            for (let j = 0; j < this.fs; j++)
                for (let k = 0; k < this.fd; k++)
                    optimizer.Update(this.w[index][i][j][k], this.batchSize)

        optimizer.Update(this.b[index], this.batchSize)
    }
}

ConvLayer.prototype.SetWeights = function(layer) {
    for (let index = 0; index < this.fc; index++) {
        for (let i = 0; i < this.fs; i++)
            for (let j = 0; j < this.fs; j++)
                for (let k = 0; k < this.fd; k++)
                    this.w[index][i][j][k] = layer.w[index][i][j][k].Copy()

        this.b[index] = layer.b[index].Copy()
    }
}
