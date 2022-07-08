function DeepQNetwork(environment, config) {
    this.environment = environment

    this.batchSize = config.batchSize
    this.minReplaySize = Math.max(config.minReplaySize, this.batchSize)
    this.replayBuffer = new ReplayBuffer(config.maxReplaySize)
    this.optimizer = new Optimizer(config.learningRate, 0, config.optimizer)
    this.loss = new Huber(1)

    this.ddqn = config.ddqn
    this.model = this.InitAgent(config.agentArchitecture)
    this.targetModel = this.InitAgent(config.agentArchitecture)
    this.targetModel.SetWeights(this.model)

    this.maxEpsilon = config.maxEpsilon
    this.minEpsilon = config.minEpsilon
    this.decay = config.decay
    this.alpha = config.alpha
    this.gamma = config.gamma

    this.trainModelPeriod = config.trainModelPeriod
    this.updateTargetModelPeriod = config.updateTargetModelPeriod
}

DeepQNetwork.prototype.InitAgent = function(architecture) {
    let inputs = this.environment.observationSpace.GetShape()
    let outputs = this.environment.actionSpace.GetShape()
    let agent = new NeuralNetwork(inputs)

    for (let layer of architecture)
        agent.AddLayer(layer)

    agent.AddLayer({type: 'fc', size: outputs, activation: ''})
    agent.SetBatchSize(this.batchSize)

    return agent
}

DeepQNetwork.prototype.GetAction = function(state, epsilon) {
    if (Math.random() <= epsilon)
        return this.environment.actionSpace.Sample()

    return Argmax(this.model.Predict(state))
}

DeepQNetwork.prototype.Train = function() {
    if (this.replayBuffer.Length() < this.minReplaySize)
        return

    let miniBatch = this.replayBuffer.Sample(this.batchSize)

    let currStates = miniBatch.map((v) => v.state)
    let nextStates = miniBatch.map((v) => v.nextState)

    let targetVal = this.targetModel.Forward(nextStates)
    let targetNext = this.ddqn ? this.model.Forward(nextStates) : null
    let target = this.model.Forward(currStates)

    let targetQs = []

    for (let index = 0; index < this.batchSize; index++) {
        let info = miniBatch[index]
        let targetQ = target[index].slice()
        targetQ[info.action] = info.reward

        if (!info.done) {
            if (this.ddqn) {
                targetQ[info.action] += this.gamma * targetVal[index][Argmax(targetNext[index])]
            }
            else {
                targetQ[info.action] += this.gamma * Max(targetVal[index])
            }
        }

        targetQ[info.action] = (1 - this.alpha) * target[index][info.action] + this.alpha * targetQ[info.action]
        targetQs.push(targetQ)
    }

    this.model.TrainOnBatchWithOutput(currStates, targetQs, target, this.optimizer, this.loss)
}

DeepQNetwork.prototype.Reset = function() {
    this.done = false
    this.episode = 0
    this.replayBuffer.Clear()
    this.epsilon = this.maxEpsilon
    this.totalTrainingRewards = 0
    this.stepsToUpdateTargetModel = 0
    this.state = this.environment.Reset(true)
}

DeepQNetwork.prototype.Step = function() {
    let action = this.GetAction(this.state, this.epsilon)
    let step = this.environment.Step(action)

    this.replayBuffer.Add(this.state, action, step.reward, step.state, step.done)
    this.done = step.done

    if (++this.stepsToUpdateTargetModel % this.trainModelPeriod == 0 || this.done)
        this.Train()

    this.state = step.state
    this.totalTrainingRewards += step.reward

    if (!this.done)
        return { done: false }

    if (this.stepsToUpdateTargetModel >= this.updateTargetModelPeriod) {
        this.targetModel.SetWeights(this.model)
        this.stepsToUpdateTargetModel = 0
    }

    return { done: true, reward: this.totalTrainingRewards }
}

DeepQNetwork.prototype.ResetEpisode = function() {
    this.epsilon = this.minEpsilon + (this.maxEpsilon - this.minEpsilon) * Math.exp(-this.decay * this.episode)
    this.episode++

    this.state = this.environment.Reset()
    this.done = false
    this.totalTrainingRewards = 0
}
