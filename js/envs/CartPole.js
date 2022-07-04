function CartPole(canvas, infoBox) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.infoBox = infoBox

    this.gravity = 9.8
    this.cartMass = 1.0
    this.poleMass = 0.1
    this.totalMass = this.poleMass + this.cartMass
    this.length = 0.5
    this.polemassLength = this.poleMass * this.length
    this.forceMag = 10
    this.tau = 0.02
    this.thetaThreshold = 12 * Math.PI / 180
    this.xThreshold = 2.4
    this.inf = 1000

    let low = [-this.xThreshold * 2, -this.inf, -this.thetaThreshold * 2, -this.inf]
    let high = [this.xThreshold * 2, this.inf, this.thetaThreshold * 2, this.inf]

    this.actionSpace = new DiscreteSpace(2)
    this.observationSpace = new BoxSpace(low, high)

    this.state = null
    this.maxSteps = 0
}

CartPole.prototype.StateToVector = function() {
    return [this.state.x, this.state.xDot, this.state.theta, this.state.thetaDot]
}

CartPole.prototype.Reset = function() {
    this.state = {
        x: RandomUniform(-0.05, 0.05),
        xDot: RandomUniform(-0.05, 0.05),
        theta: RandomUniform(-0.05, 0.05),
        thetaDot: RandomUniform(-0.05, 0.05),
    }
    this.steps = 0

    return this.StateToVector()
}

CartPole.prototype.Step = function(action) {
    if (!this.actionSpace.Contains(action))
        throw new Error(`Action "${action}" is invalid`)

    if (this.state == null)
        throw new Error(`State is null. Call reset before using step method`)

    let force = action == 1 ? this.forceMag : -this.forceMag
    let cosTheta = Math.cos(this.state.theta)
    let sinTheta = Math.sin(this.state.theta)

    let tmp = (force + this.polemassLength * this.state.thetaDot * this.state.thetaDot * sinTheta) / this.totalMass
    let thetaAcc = (this.gravity * sinTheta - cosTheta * tmp) / (this.length * (4/3 - this.poleMass * cosTheta * cosTheta / this.totalMass))
    let xAcc = tmp - this.polemassLength * thetaAcc * cosTheta / this.totalMass

    this.state.x += this.tau * this.state.xDot
    this.state.xDot += this.tau * xAcc
    this.state.theta += this.tau * this.state.thetaDot
    this.state.thetaDot += this.tau * thetaAcc
    this.steps++
    this.maxSteps = Math.max(this.steps, this.maxSteps)

    let done = Math.abs(this.state.x) > this.xThreshold || Math.abs(this.state.theta) > this.thetaThreshold
    let reward = 1 - (Math.abs(this.state.x) / this.xThreshold + Math.abs(this.state.theta) / this.thetaThreshold) / 2

    return {
        state: this.StateToVector(),
        reward: reward,
        done: done
    }
}

CartPole.prototype.Draw = function() {
    if (this.state == null)
        return null

    let width = this.canvas.width
    let height = this.canvas.height
    let worldWidth = this.xThreshold * 2
    let scale = width / worldWidth

    let poleWidth = 10
    let poleLen = scale * 2 * this.length

    let cartX = this.state.x * scale + width / 2
    let cartY = height - 100
    let cartWidth = 50
    let cartHeight = 30

    let axlOffset = cartHeight / 4

    this.ctx.clearRect(0, 0, width, height)
    this.ctx.fillStyle = '#000'
    this.ctx.fillRect(cartX - cartWidth / 2, cartY - cartHeight / 2, cartWidth, cartHeight)

    this.ctx.beginPath()
    this.ctx.strokeStyle = '#000'
    this.ctx.moveTo(0, cartY)
    this.ctx.lineTo(width, cartY)
    this.ctx.stroke()

    this.ctx.save()
    this.ctx.translate(cartX, cartY - axlOffset)
    this.ctx.rotate(this.state.theta)
    this.ctx.fillStyle = '#ca9865'
    this.ctx.fillRect(-poleWidth / 2, -poleLen, poleWidth, poleLen)
    this.ctx.restore()

    this.ctx.beginPath()
    this.ctx.fillStyle = '#8184cb'
    this.ctx.arc(cartX, cartY - axlOffset, poleWidth / 2, 0, Math.PI * 2)
    this.ctx.fill()

    this.ctx.font = '16px sans-serif'
    this.ctx.textAlign = 'left'
    this.ctx.textBaseline = 'bottom'
    this.ctx.fillStyle = '#888'
    this.ctx.fillText(`x: ${this.state.x.toFixed(3)}`, 5, height - 25)
    this.ctx.fillText(`theta: ${this.state.theta.toFixed(3)}`, 5, height - 5)

    this.infoBox.innerText = `Текущее число шагов: ${this.steps}\n`
    this.infoBox.innerText += `Максимальное число шагов: ${this.maxSteps}\n`
}
