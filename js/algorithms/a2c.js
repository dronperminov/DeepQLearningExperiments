function AdvancedActorCritic(environment, config) {
    this.environment = environment

    this.actor = this.InitActor(config.actorArchitecture)
    this.critic = this.InitCritic(config.criticArchitecture)
    this.optimizer = new Optimizer(config.learningRate, 0, config.optimizer)
    this.gamma = config.gamma
}

AdvancedActorCritic.prototype.InitActor = function(architecture) {
    let inputs = this.environment.GetObservationShape()
    let outputs = this.environment.actionSpace.GetShape()
    let actor = new NeuralNetwork(inputs)

    for (let layer of architecture)
        actor.AddLayer(layer)

    actor.AddLayer({type: 'fc', size: outputs, activation: ''})
    actor.AddLayer({type: 'softmax'})

    return actor
}

AdvancedActorCritic.prototype.InitCritic = function(architecture) {
    let inputs = this.environment.GetObservationShape()
    let critic = new NeuralNetwork(inputs)

    for (let layer of architecture)
        critic.AddLayer(layer)

    critic.AddLayer({type: 'fc', size: 1, activation: ''})

    return critic
}

AdvancedActorCritic.prototype.GetAction = function(state) {
    let probs = this.actor.Predict(state)
    let critic = this.critic.Predict(state)
    let action = this.environment.actionSpace.ProbabilitySample(probs)
    let prob = probs[action]

    return {action, prob, critic}
}

AdvancedActorCritic.prototype.GetDiscountedRewards = function() {
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

AdvancedActorCritic.prototype.UpdatePolicy = function() {
    let discountedRewards = this.GetDiscountedRewards()

    let actorDeltas = []
    let criticDeltas = []

    for (let i = 0; i < discountedRewards.length; i++) {
        let actorDelta = []

        for (let j = 0; j < this.environment.actionSpace.shape; j++)
            actorDelta[j] = 0

        actorDelta[this.actions[i]] = (this.critics[i] - discountedRewards[i]) / this.probs[i]
        actorDeltas.push(actorDelta)

        let criticDelta = this.critics[i] - discountedRewards[i]

        if (Math.abs(criticDelta) > 1)
            criticDelta = Math.sign(criticDelta)

        criticDeltas.push([criticDelta])
    }

    this.actor.SetBatchSize(this.states.length)
    this.actor.Forward(this.states)
    this.actor.ZeroGradients()
    this.actor.Backward(this.states, actorDeltas)
    this.actor.UpdateWeights(this.optimizer)

    this.critic.SetBatchSize(this.states.length)
    this.critic.Forward(this.states)
    this.critic.ZeroGradients()
    this.critic.Backward(this.states, criticDeltas)
    this.critic.UpdateWeights(this.optimizer)
}

AdvancedActorCritic.prototype.Reset = function() {
    this.environment.ResetInfo()
    this.ResetEpisode()
}

AdvancedActorCritic.prototype.Step = function() {
    let {action, prob, critic} = this.GetAction(this.state)
    let step = this.environment.Step(action)

    this.critics.push(critic)
    this.probs.push(prob)
    this.rewards.push(step.reward)
    this.actions.push(action)
    this.states.push(this.state)

    this.state = step.state
    this.done = step.done
    this.totalTrainingRewards += step.reward

    if (!this.done)
        return {done: false}

    this.UpdatePolicy()

    return { done: true, reward: this.totalTrainingRewards }
}

AdvancedActorCritic.prototype.ResetEpisode = function() {
    this.done = false
    this.totalTrainingRewards = 0
    this.state = this.environment.Reset()

    this.probs = []
    this.critics = []
    this.rewards = []
    this.actions = []
    this.states = []
}