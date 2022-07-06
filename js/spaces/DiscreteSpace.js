function DiscreteSpace(count, start = 0) {
    this.count = count
    this.start = start
    this.shape = count
}

DiscreteSpace.prototype.Sample = function() {
    return Math.floor(this.start + Math.random() * this.count)
}

DiscreteSpace.prototype.ProbabilitySample = function(probabilities) {
    let p = Math.random()
    let sum = 0
    let last = this.count - 1

    for (let i = 0; i < last; i++) {
        sum += probabilities[i]

        if (p < sum)
            return this.start + i
    }

    return this.start + last
}

DiscreteSpace.prototype.GetShape = function() {
    return this.shape
}

DiscreteSpace.prototype.Contains = function(action) {
    return this.start <= action && action <= this.start + this.count
}