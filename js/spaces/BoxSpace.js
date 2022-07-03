function BoxSpace(low, high) {
    this.low = low
    this.high = high

    if (low.length != high.length)
        throw new Error("Sizes of low and high are not equal")

    this.shape = low.length
}

BoxSpace.prototype.Sample = function() {
    let sample = []

    for (let i = 0; i < this.shape; i++)
        sample[i] = RandomUniform(this.low[i], this.high[i])

    return sample
}

BoxSpace.prototype.GetShape = function() {
    return this.shape
}
