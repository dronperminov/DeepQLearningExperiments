function ReplayBuffer(maxSize = 10000) {
    this.maxSize = maxSize
    this.Clear()
}

ReplayBuffer.prototype.Add = function(state, action, reward, nextState, done) {
    let replay = {state, action, reward, nextState, done}

    if (this.buffer.length < this.maxSize) {
        this.buffer.push(replay)
    }
    else {
        this.buffer[this.position] = replay
        this.position = (this.position + 1) % this.maxSize
    }
}

ReplayBuffer.prototype.Clear = function() {
    this.position = 0
    this.buffer = []
}

ReplayBuffer.prototype.Length = function() {
    return this.buffer.length
}

ReplayBuffer.prototype.Sample = function(batchSize) {
    let sample = []

    for (let i = 0; i < batchSize; i++) {
        let index = Math.floor(Math.random() * this.buffer.length)
        sample.push(this.buffer[index])
    }

    return sample
}
