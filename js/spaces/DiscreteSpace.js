function DiscreteSpace(count, start = 0) {
    this.count = count
    this.start = start
    this.shape = count
}

DiscreteSpace.prototype.Sample = function() {
    return Math.floor(this.start + Math.random() * this.count)
}

DiscreteSpace.prototype.GetShape = function() {
    return this.shape
}

DiscreteSpace.prototype.Contains = function(action) {
    return this.start <= action && action <= this.start + this.count
}