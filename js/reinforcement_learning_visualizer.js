function ReinforcementLearningVisualizer(algorithm, rewardCanvas, runBtn, stepBtn, resetBtn) {
    this.algorithm = algorithm

    this.rewardCanvas = rewardCanvas
    this.rewardCtx = this.rewardCanvas.getContext('2d')

    this.runBtn = runBtn
    this.stepBtn = stepBtn
    this.resetBtn = resetBtn

    this.runBtn.addEventListener('click', () => this.StartStop())
    this.stepBtn.addEventListener('click', () => { this.Stop(); this.Step() })
    this.resetBtn.addEventListener('click', () => this.Reset())

    this.Reset()
    this.StepAnimation()
}

ReinforcementLearningVisualizer.prototype.DrawRewards = function(rewards, minRewards = 10) {
    let width = this.rewardCanvas.width
    let height = this.rewardCanvas.height
    let padding = 15

    this.rewardCtx.clearRect(0, 0, width, height)

    if (rewards.length == 0)
        return

    let count = Math.max(rewards.length, minRewards)
    let maxReward = -Infinity
    let minReward = Infinity

    for (let reward of rewards) {
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

ReinforcementLearningVisualizer.prototype.Reset = function() {
    this.isRun = false
    this.algorithm.Reset()
    this.Stop()
    this.DrawRewards([])
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
    if (!this.algorithm.Step())
        return

    this.DrawRewards(this.algorithm.rewards)
}

ReinforcementLearningVisualizer.prototype.StepAnimation = function() {
    if (this.isRun)
        this.Step()

    window.requestAnimationFrame(() => this.StepAnimation())
}
