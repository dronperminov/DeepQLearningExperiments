function DeepQNetwork(environment, environmentInfoBox) {
    this.environment = environment
    this.environmentInfoBox = environmentInfoBox

    this.batchSize = 128
    this.optimizer = new Optimizer(0.1, 0, 'sgd')
    this.loss = new Huber(1)

    this.model = this.InitAgent()
    this.targetModel = this.InitAgent()
    this.targetModel.SetWeights(this.model)

    this.maxEpsilon = 1
    this.minEpsilon = 0.01
    this.decay = 0.01
}

DeepQNetwork.prototype.InitAgent = function() {
    let inputs = this.environment.observationSpace.GetShape()
    let outputs = this.environment.actionSpace.GetShape()

    let agent = new NeuralNetwork(inputs)
    agent.AddLayer({size: 24, activation: 'relu'})
    agent.AddLayer({size: 12, activation: 'relu'})
    agent.AddLayer({size: outputs, activation: ''})
    agent.SetBatchSize(this.batchSize)

    return agent
}

DeepQNetwork.prototype.GetMaxValue = function(values) {
    let maxValue = values[0]

    for (let value of values)
        if (value > maxValue)
            maxValue = value

    return maxValue
}

DeepQNetwork.prototype.Average = function(values) {
    let average = 0

    for (let value of values)
        average += value

    return average / values.length
}

DeepQNetwork.prototype.Train = function(replayMemory) {
    let learningRate = 0.7
    let discountFactor = 0.618

    if (replayMemory.length < this.batchSize)
        return

    let miniBatch = RandomSample(replayMemory, this.batchSize)

    let currentStates = miniBatch.map((v) => v.observation)
    let currentQsList = this.model.Forward(currentStates)
    
    let newCurrentStates = miniBatch.map((v) => v.newObservation)
    let newQsList = this.targetModel.Forward(newCurrentStates)

    let targetQsList = []

    for (let index = 0; index < this.batchSize; index++) {
        let info = miniBatch[index]
        let maxFutureQ = info.reward

        if (!info.done)
            maxFutureQ += discountFactor * this.GetMaxValue(newQsList[index])

        let targetQ = currentQsList[index].slice()
        targetQ[info.action] = (1 - learningRate) * targetQ[info.action] + learningRate * maxFutureQ
        targetQsList.push(targetQ)
    }

    this.model.TrainOnBatch(currentStates, targetQsList, this.optimizer, this.loss)
}

DeepQNetwork.prototype.Step = function(trainSteps, episode, episodeInfo, epsilon, replayMemory, stepsToUpdateTargetModel, totalTrainingRewards, observation, done) {
    if (episode >= trainSteps)
        return

    if (!done) {
        episodeInfo.length++
        episodeInfo.maxLength = Math.max(episodeInfo.length, episodeInfo.maxLength)
        stepsToUpdateTargetModel++

        this.environmentInfoBox.innerText = `Текущее число шагов: ${episodeInfo.length}\n`
        this.environmentInfoBox.innerText += `Максимальное число шагов: ${episodeInfo.maxLength}\n`
        this.environmentInfoBox.innerText += `Среднее число шагов за последние ${episodeInfo.avgLength.length} шагов: ${Math.round(this.Average(episodeInfo.avgLength))}`
        this.environment.Draw()

        let action

        if (Math.random() <= epsilon) {
            action = this.environment.actionSpace.Sample()
        }
        else {
            action = this.model.PredictArgmax(observation)
        }

        let step = this.environment.Step(action)
        replayMemory.push({
            observation: observation,
            action: action,
            reward: step.reward,
            newObservation: step.state,
            done: step.done
        })

        if (stepsToUpdateTargetModel % 4 == 0 || done)
            this.Train(replayMemory)

        observation = step.state
        totalTrainingRewards += step.reward
        done = step.done
    }
    
    if (done) {
        console.log(`${episode}. Total training rewards: ${totalTrainingRewards} use ${episodeInfo.length} steps`)

        if (stepsToUpdateTargetModel >= 100) {
            console.log("Copying main network weights to target network")
            this.targetModel.SetWeights(this.model)
            stepsToUpdateTargetModel = 0
        }

        epsilon = this.minEpsilon + (this.maxEpsilon - this.minEpsilon) * Math.exp(-this.decay * episode)
        episode++

        done = false
        totalTrainingRewards = 0
        observation = this.environment.Reset()
        episodeInfo.avgLength[episode % episodeInfo.avgLength.length] = episodeInfo.length
        episodeInfo.length = 0
    }

    window.requestAnimationFrame(() => this.Step(trainSteps, episode, episodeInfo, epsilon, replayMemory, stepsToUpdateTargetModel, totalTrainingRewards, observation, done))
}

DeepQNetwork.prototype.Run = function(trainSteps = 10000) {
    let observation = this.environment.Reset()
    let episodeInfo = {
        length: 0,
        maxLength: 0,
        avgLength: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    this.Step(trainSteps, 0, episodeInfo, 1, [], 0, 0, observation, false)
}
