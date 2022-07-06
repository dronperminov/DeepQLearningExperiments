function DeepQNetwork(environment, rewardCanvas, config) {
    this.environment = environment
    this.rewardCanvas = rewardCanvas
    this.rewardCtx = this.rewardCanvas.getContext('2d')

    this.batchSize = config.batchSize
    this.minReplaySize = Math.max(config.minReplaySize, this.batchSize)
    this.replayBuffer = new ReplayBuffer(config.maxReplaySize)
    this.optimizer = new Optimizer(config.learningRate, 0, config.optimizer)
    this.loss = new Huber(1)

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

DeepQNetwork.prototype.DrawRewards = function(rewards, minRewards = 10) {
    let width = this.rewardCanvas.width
    let height = this.rewardCanvas.height
    let padding = 15

    let count = Math.max(rewards.length, minRewards)
    let maxReward = -Infinity
    let minReward = Infinity

    for (let reward of rewards) {
        maxReward = Math.max(maxReward, reward)
        minReward = Math.min(minReward, reward)
    }

    this.rewardCtx.clearRect(0, 0, width, height)

    this.rewardCtx.strokeStyle = '#000'
    this.rewardCtx.beginPath()
    this.rewardCtx.moveTo(padding, padding)
    this.rewardCtx.lineTo(padding, height - padding)
    this.rewardCtx.lineTo(width - padding, height - padding)
    this.rewardCtx.stroke()

    this.rewardCtx.fillStyle = '#000'
    this.rewardCtx.font = '10px sans-serif'
    this.rewardCtx.textAlign = 'center'
    this.rewardCtx.textBaseline = 'bottom'
    this.rewardCtx.fillText(`${maxReward.toFixed(2)}`, padding, padding)
    this.rewardCtx.textBaseline = 'top'
    this.rewardCtx.fillText(`${minReward.toFixed(2)}`, padding, height - padding + 2)
    this.rewardCtx.textAlign = 'right'
    this.rewardCtx.fillText(`${rewards.length}`, padding + (rewards.length - 1) / (count - 1) * (width - 2 * padding), height - padding + 2)

    this.rewardCtx.strokeStyle = '#f00'
    this.rewardCtx.beginPath()

    for (let i = 0; i < rewards.length; i++) {
        let x = padding + i / (count - 1) * (width - 2 * padding)
        let y = height - padding - ((rewards[i] - minReward) / (maxReward - minReward)) * (height - 2 * padding)

        if (i == 0)
            this.rewardCtx.moveTo(x, y)
        else
            this.rewardCtx.lineTo(x, y)
    }

    this.rewardCtx.stroke()
}

DeepQNetwork.prototype.Train = function() {
    if (this.replayBuffer.Length() < this.minReplaySize)
        return

    let miniBatch = this.replayBuffer.Sample(this.batchSize)

    let currStates = miniBatch.map((v) => v.state)
    let nextStates = miniBatch.map((v) => v.nextState)

    let currQs = this.model.Forward(currStates)
    let nextQs = this.targetModel.Forward(nextStates)

    let targetQs = []

    for (let index = 0; index < this.batchSize; index++) {
        let info = miniBatch[index]
        let maxFutureQ = info.reward

        if (!info.done)
            maxFutureQ += this.gamma * Max(nextQs[index])

        let targetQ = currQs[index].slice()
        targetQ[info.action] = (1 - this.alpha) * targetQ[info.action] + this.alpha * maxFutureQ
        targetQs.push(targetQ)
    }

    this.model.TrainOnBatchWithOutput(currStates, targetQs, currQs, this.optimizer, this.loss)
}

DeepQNetwork.prototype.Step = function(trainSteps, episode, epsilon, stepsToUpdateTargetModel, totalTrainingRewards, state, done, rewards) {
    if (episode >= trainSteps)
        return

    if (!done) {
        stepsToUpdateTargetModel++

        this.environment.Draw()

        let action = this.GetAction(epsilon)
        let step = this.environment.Step(action)

        this.replayBuffer.Add(state, action, step.reward, step.state, step.done)

        if (stepsToUpdateTargetModel % this.trainModelPeriod == 0 || done)
            this.Train()

        state = step.state
        totalTrainingRewards += step.reward
        done = step.done
    }

    if (done) {
        rewards.push(totalTrainingRewards)
        this.DrawRewards(rewards)

        console.log(`${episode}. Total training rewards: ${totalTrainingRewards} use ${this.environment.steps} steps`)

        if (stepsToUpdateTargetModel >= this.updateTargetModelPeriod) {
            console.log("Copying main network weights to target network")
            this.targetModel.SetWeights(this.model)
            stepsToUpdateTargetModel = 0
        }

        epsilon = this.minEpsilon + (this.maxEpsilon - this.minEpsilon) * Math.exp(-this.decay * episode)
        episode++

        done = false
        totalTrainingRewards = 0
        state = this.environment.Reset()
    }

    window.requestAnimationFrame(() => this.Step(trainSteps, episode, epsilon, stepsToUpdateTargetModel, totalTrainingRewards, state, done, rewards))
}

DeepQNetwork.prototype.Run = function(trainSteps = Infinity) {
    let state = this.environment.Reset()

    this.Step(trainSteps, 0, 1, 0, 0, state, false, [])
}
