function ReinforcementLearningVisualizer(algorithm, environmentCanvas, environmentInfoBox, rewardCanvas, runBtn, stepBtn, stepEpisodeBtn, resetBtn) {
    this.algorithm = algorithm
    this.environment = algorithm.environment

    this.environmentCanvas = environmentCanvas
    this.environmentCtx = environmentCanvas.getContext('2d')
    this.environmentInfoBox = environmentInfoBox

    this.rewardCanvas = rewardCanvas
    this.rewardCtx = this.rewardCanvas.getContext('2d')
    this.rewards = []

    this.runBtn = runBtn
    this.stepBtn = stepBtn
    this.stepEpisodeBtn = stepEpisodeBtn
    this.resetBtn = resetBtn

    this.runBtn.addEventListener('click', () => this.StartStop())
    this.stepBtn.addEventListener('click', () => { this.Stop(); this.Step() })
    this.stepEpisodeBtn.addEventListener('click', () => { this.Stop(); this.StepEpsiode() })
    this.resetBtn.addEventListener('click', () => this.Reset())

    this.Reset()
    this.StepAnimation()
}

ReinforcementLearningVisualizer.prototype.DrawEnvironment = function() {
    this.environment.Draw(this.environmentCtx, this.environmentInfoBox)
}

ReinforcementLearningVisualizer.prototype.DrawRewards = function(minRewards = 10) {
    let width = this.rewardCanvas.width
    let height = this.rewardCanvas.height
    let padding = 15

    this.rewardCtx.clearRect(0, 0, width, height)

    if (this.rewards.length == 0)
        return

    let count = Math.max(this.rewards.length, minRewards)
    let maxReward = -Infinity
    let minReward = Infinity

    for (let reward of this.rewards) {
        maxReward = Math.max(maxReward, reward)
        minReward = Math.min(minReward, reward)
    }

    this.rewardCtx.strokeStyle = '#000'
    this.rewardCtx.beginPath()
    this.rewardCtx.moveTo(padding, padding)
    this.rewardCtx.lineTo(padding, height - padding)
    this.rewardCtx.lineTo(width - padding, height - padding)
    this.rewardCtx.stroke()

    this.rewardCtx.fillStyle = '#000'
    this.rewardCtx.font = `${Math.min(width / 20, 12)}px sans-serif`
    this.rewardCtx.textAlign = 'left'
    this.rewardCtx.textBaseline = 'bottom'
    this.rewardCtx.fillText(`${maxReward.toFixed(2)}`, 2, padding)
    this.rewardCtx.textBaseline = 'top'
    this.rewardCtx.fillText(`${minReward.toFixed(2)}`, 2, height - padding + 2)
    this.rewardCtx.textAlign = 'right'
    this.rewardCtx.fillText(`${this.rewards.length}`, padding + (this.rewards.length - 1) / (count - 1) * (width - 2 * padding), height - padding + 2)

    this.rewardCtx.strokeStyle = '#f00'
    this.rewardCtx.beginPath()

    for (let i = 0; i < this.rewards.length; i++) {
        let x = padding + i / (count - 1) * (width - 2 * padding)
        let y = height - padding - ((this.rewards[i] - minReward) / (maxReward - minReward)) * (height - 2 * padding)

        if (i == 0)
            this.rewardCtx.moveTo(x, y)
        else
            this.rewardCtx.lineTo(x, y)
    }

    this.rewardCtx.stroke()
}

ReinforcementLearningVisualizer.prototype.Reset = function() {
    this.Stop()
    this.algorithm.Reset()
    this.DrawEnvironment()
    this.DrawRewards()
}

ReinforcementLearningVisualizer.prototype.Start = function() {
    this.isRun = true
    this.runBtn.value = 'Остановить'
}

ReinforcementLearningVisualizer.prototype.Stop = function() {
    this.isRun = false
    this.runBtn.value = 'Запустить'
}

ReinforcementLearningVisualizer.prototype.StartStop = function() {
    if (this.isRun) {
        this.Stop()
    }
    else {
        this.Start()
    }
}

ReinforcementLearningVisualizer.prototype.Step = function() {
    let step = this.algorithm.Step()
    this.DrawEnvironment()

    if (step.done) {
        this.rewards.push(step.reward)
        this.algorithm.ResetEpisode()
        this.DrawRewards()
    }

    return step.done
}

ReinforcementLearningVisualizer.prototype.StepEpsiode = function() {
    if (this.Step())
        return

    window.requestAnimationFrame(() => this.StepEpsiode())
}

ReinforcementLearningVisualizer.prototype.StepAnimation = function() {
    if (this.isRun)
        this.Step()

    window.requestAnimationFrame(() => this.StepAnimation())
}
