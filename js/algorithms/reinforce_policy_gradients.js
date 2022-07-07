function ReinforcePolicyGradients(environment, config) {
    this.environment = environment

    this.agent = this.InitAgent(config.agentArchitecture)
    this.optimizer = new Optimizer(config.learningRate, 0, config.optimizer)
    this.gamma = config.gamma
}

ReinforcePolicyGradients.prototype.InitAgent = function(architecture) {
    let inputs = this.environment.observationSpace.GetShape()
    let outputs = this.environment.actionSpace.GetShape()
    let agent = new NeuralNetwork(inputs)

    for (let layer of architecture)
        agent.AddLayer(layer)

    agent.AddLayer({type: 'fc', size: outputs, activation: ''})
    agent.AddLayer({type: 'softmax'})

    return agent
}

ReinforcePolicyGradients.prototype.GetAction = function(state) {
    let probs = this.agent.Predict(state)
    let action = this.environment.actionSpace.ProbabilitySample(probs)

    return {action, probs}
}

ReinforcePolicyGradients.prototype.GetDiscountedRewards = function() {
    let discountedRewards = []
    let discountedReward = 0

    for (let i = this.rewards.length - 1; i >= 0; i--) {
        discountedReward = this.rewards[i] + this.gamma * discountedReward
        discountedRewards.push(discountedReward)
    }

    discountedRewards.reverse()
    Standartize(discountedRewards)

    return discountedRewards
}

ReinforcePolicyGradients.prototype.UpdatePolicy = function() {
    let discountedRewards = this.GetDiscountedRewards()
    let deltas = []

    for (let i = 0; i < discountedRewards.length; i++) {
        let delta = []

        for (let j = 0; j < this.environment.actionSpace.shape; j++)
            delta[j] = 0

        delta[this.actions[i]] = -discountedRewards[i] / this.probs[i][this.actions[i]] * this.states.length
        deltas.push(delta)
    }

    this.agent.SetBatchSize(this.states.length)
    this.agent.Forward(this.states)
    this.agent.ZeroGradients()
    this.agent.Backward(this.states, deltas)
    this.agent.UpdateWeights(this.optimizer)
}

ReinforcePolicyGradients.prototype.Reset = function() {
    this.done = false
    this.totalTrainingRewards = 0
    this.state = this.environment.Reset(true)

    this.probs = []
    this.states = []
    this.rewards = []
    this.actions = []
}

ReinforcePolicyGradients.prototype.Step = function() {
    let {action, probs} = this.GetAction(this.state)
    let step = this.environment.Step(action)

    this.probs.push(probs)
    this.states.push(this.state)
    this.rewards.push(step.reward)
    this.actions.push(action)

    this.state = step.state
    this.done = step.done
    this.totalTrainingRewards += step.reward

    if (!this.done)
        return { done: false }

    this.UpdatePolicy()

    return { done: true, reward: this.totalTrainingRewards }
}

ReinforcePolicyGradients.prototype.ResetEpisode = function() {
    this.done = false
    this.totalTrainingRewards = 0
    this.state = this.environment.Reset()

    this.probs = []
    this.states = []
    this.rewards = []
    this.actions = []
}
