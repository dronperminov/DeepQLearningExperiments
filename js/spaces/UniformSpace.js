function UniformSpace(low, high, count) {
    this.low = low
    this.high = high
    this.shape = count
}

UniformSpace.prototype.Sample = function() {
    let sample = []

    for (let i = 0; i < this.shape; i++)
        sample[i] = RandomUniform(this.low, this.high)

    return sample
}

UniformSpace.prototype.GetShape = function() {
    return this.shape
}
