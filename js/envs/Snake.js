const SNAKE_INITIAL_LENGTH = 3

const SNAKE_TURN_LEFT = 1
const SNAKE_TURN_RIGHT = 2

const SNAKE_EAT_SELF = 'eat self'
const SNAKE_WALL = 'wall'
const SNAKE_EAT_FOOD = 'eat food'
const SNAKE_NO_EAT = 'no eat'
const SNAKE_DEFAULT = 'default'

function Snake(fieldWidth = 14, fieldHeight = 9) {
    this.fieldWidth = fieldWidth
    this.fieldHeight = fieldHeight

    this.snake = null
    this.food = null
    this.direction = null

    this.actionSpace = new DiscreteSpace(3)
    this.observationSpace = new UniformSpace(-1, 1, 43)

    this.ResetInfo()
}

Snake.prototype.InitSnake = function() {
    let snake = []

    let x0 = Math.floor(this.fieldWidth / 2)
    let y0 = Math.floor(this.fieldHeight / 2)

    for (let i = 0; i < SNAKE_INITIAL_LENGTH; i++)
        snake.push({x: x0, y: y0 + i})

    return snake
}

Snake.prototype.IsInsideSnake = function(x, y, start = 0) {
    let snake = this.snake

    for (let i = start; i < snake.length; i++)
        if (snake[i].x == x && snake[i].y == y)
            return true

    return false
}

Snake.prototype.InitFood = function() {
    let food = {
        x: 0,
        y: 0
    }

    do {
        food.x = Math.floor(Math.random() * (this.fieldWidth + 1))
        food.y = Math.floor(Math.random() * (this.fieldHeight + 1))
    } while (this.IsInsideSnake(food.x, food.y))

    return food
}

Snake.prototype.MoveSnake = function(dx, dy) {
    let head = this.snake[0]

    if (head.x + dx < 0 || head.y + dy < 0 || head.x + dx > this.fieldWidth || head.y + dy > this.fieldHeight) {
        this.wall++
        return SNAKE_WALL
    }

    if (this.IsInsideSnake(head.x + dx, head.y + dy, 1)) {
        this.eatSelf++
        return SNAKE_EAT_SELF
    }

    if (head.x + dx == this.food.x && head.y + dy == this.food.y) {
        this.snake.unshift({x: this.food.x, y: this.food.y})
        this.food = this.InitFood()
        this.maxLength = Math.max(this.snake.length, this.maxLength)
        this.stepsWithoutFood = 0

        return SNAKE_EAT_FOOD
    }

    this.stepsWithoutFood++

    if (this.stepsWithoutFood > this.fieldWidth * this.fieldHeight * 2) {
        this.noEat++
        return SNAKE_NO_EAT
    }

    for (let i = this.snake.length - 1; i > 0; i--) {
        this.snake[i].x = this.snake[i - 1].x
        this.snake[i].y = this.snake[i - 1].y
    }

    this.snake[0].x += dx
    this.snake[0].y += dy

    return SNAKE_DEFAULT
}

Snake.prototype.IsCollision = function(point) {
    if (point.x < 0 || point.x > this.fieldWidth)
        return true

    if (point.y < 0 || point.y > this.fieldHeight)
        return true

    return this.IsInsideSnake(point.x, point.y)
}

Snake.prototype.DistanceToCollision = function(x0, y0, dx, dy, fromHead = false) {
    let x = x0 + (fromHead ? dx : 0)
    let y = y0 + (fromHead ? dy : 0)
    let i = 1

    while (0 <= x && x <= this.fieldWidth && 0 <= y && y <= this.fieldHeight && !this.IsInsideSnake(x, y, 1)) {
        x += dx
        y += dy
        i++
    }

    return [dx * i / this.fieldWidth, dy * i / this.fieldHeight]
}

Snake.prototype.StateToVector = function() {
    let head = this.snake[0]
    let pointL = { x: head.x - 1, y: head.y }
    let pointR = { x: head.x + 1, y: head.y }
    let pointU = { x: head.x, y: head.y - 1 }
    let pointD = { x: head.x, y: head.y + 1 }

    let dirL = this.direction.dx == -1
    let dirR = this.direction.dx == 1
    let dirU = this.direction.dy == -1
    let dirD = this.direction.dy == 1

    let distances = [
        ...this.DistanceToCollision(head.x, head.y, this.direction.dx, this.direction.dy, true),
        ...this.DistanceToCollision(head.x - this.direction.dy, head.y + this.direction.dx, this.direction.dx, this.direction.dy),
        ...this.DistanceToCollision(head.x + this.direction.dy, head.y - this.direction.dx, this.direction.dx, this.direction.dy),

        ...this.DistanceToCollision(head.x, head.y, -this.direction.dx, -this.direction.dy, true),
        ...this.DistanceToCollision(head.x - this.direction.dy, head.y + this.direction.dx, -this.direction.dx, -this.direction.dy),
        ...this.DistanceToCollision(head.x + this.direction.dy, head.y - this.direction.dx, -this.direction.dx, -this.direction.dy),

        ...this.DistanceToCollision(head.x, head.y, this.direction.dy, -this.direction.dx, true),
        ...this.DistanceToCollision(head.x + this.direction.dx, head.y + this.direction.dy, this.direction.dy, -this.direction.dx),
        ...this.DistanceToCollision(head.x - this.direction.dx, head.y - this.direction.dy, this.direction.dy, -this.direction.dx),

        ...this.DistanceToCollision(head.x, head.y, -this.direction.dy, this.direction.dx, true),
        ...this.DistanceToCollision(head.x + this.direction.dx, head.y + this.direction.dy, -this.direction.dy, this.direction.dx),
        ...this.DistanceToCollision(head.x - this.direction.dx, head.y - this.direction.dy, -this.direction.dy, this.direction.dx)
    ]

    return [
        (dirU && this.IsCollision(pointU)) ||
        (dirD && this.IsCollision(pointD)) ||
        (dirL && this.IsCollision(pointL)) ||
        (dirR && this.IsCollision(pointR)),

        (dirU && this.IsCollision(pointR)) ||
        (dirD && this.IsCollision(pointL)) ||
        (dirU && this.IsCollision(pointU)) ||
        (dirD && this.IsCollision(pointD)),

        (dirU && this.IsCollision(pointR)) ||
        (dirD && this.IsCollision(pointL)) ||
        (dirR && this.IsCollision(pointU)) ||
        (dirL && this.IsCollision(pointD)),

        dirL,
        dirR,
        dirU,
        dirD,

        this.food.x < head.x,
        this.food.x > head.x,
        this.food.y < head.y,
        this.food.y > head.y,

        this.direction.dx,
        this.direction.dy,

        (head.x - 0) / this.fieldWidth,
        (head.y - 0) / this.fieldHeight,
        (head.x - this.fieldWidth) / this.fieldWidth,
        (head.y - this.fieldHeight) / this.fieldHeight,

        (head.x - this.food.x) / this.fieldWidth,
        (head.y - this.food.y) / this.fieldHeight,
        
        ...distances
    ]
}

Snake.prototype.ResetInfo = function() {
    this.maxLength = SNAKE_INITIAL_LENGTH
    this.wall = 0
    this.eatSelf = 0
    this.noEat = 0
}

Snake.prototype.Reset = function() {
    this.snake = this.InitSnake()
    this.food = this.InitFood()
    this.direction = {
        dx: 0,
        dy: -1
    }

    this.stepsWithoutFood = 0

    return this.StateToVector()
}

Snake.prototype.GetReward = function(move) {
    if (move == SNAKE_WALL)
        return -100

    if (move == SNAKE_EAT_SELF)
        return -200

    if (move == SNAKE_NO_EAT)
        return -400

    if (move == SNAKE_EAT_FOOD)
        return 30

    let prevDx = this.food.x - (this.snake[0].x - this.direction.dx)
    let prevDy = this.food.y - (this.snake[0].y - this.direction.dy)
    let prevDst = Math.abs(prevDx) + Math.abs(prevDy)

    let currDx = this.food.x - this.snake[0].x
    let currDy = this.food.y - this.snake[0].y
    let currDst = Math.abs(currDx) + Math.abs(currDy)

    if (currDst < prevDst)
        return 0.5 / this.snake.length

    return -1 / this.snake.length
}

Snake.prototype.Step = function(action) {
    if (!this.actionSpace.Contains(action))
        throw new Error(`Action "${action}" is invalid`)

    if (this.snake == null)
        throw new Error(`State is null. Call reset before using step method`)

    let dx = this.direction.dx
    let dy = this.direction.dy

    if (action == SNAKE_TURN_LEFT) {
        this.direction.dx = dy
        this.direction.dy = -dx
    }
    else if (action == SNAKE_TURN_RIGHT) {
        this.direction.dx = -dy
        this.direction.dy = dx
    }

    let move = this.MoveSnake(this.direction.dx, this.direction.dy)
    let done = move == SNAKE_WALL || move == SNAKE_EAT_SELF || move == SNAKE_NO_EAT

    return {
        state: this.StateToVector(),
        reward: this.GetReward(move),
        done: done
    }
}

Snake.prototype.DrawCells = function(ctx, cellWidth, cellHeight) {
    ctx.strokeStyle = '#ccc'
    ctx.beginPath()

    for (let i = 0; i <= this.fieldHeight; i++)
        for (let j = 0; j <= this.fieldWidth; j++)
            ctx.rect(j * cellWidth, i * cellHeight, cellWidth, cellHeight)

    ctx.stroke()
}

Snake.prototype.DrawSnake = function(ctx, cellWidth, cellHeight) {
    for (let p of this.snake) {
        ctx.fillStyle = p == this.snake[0] ? '#009688' : '#4caf50'
        ctx.beginPath()
        ctx.rect(p.x * cellWidth, p.y * cellHeight, cellWidth, cellHeight)
        ctx.fill()
        ctx.stroke()
    }
}

Snake.prototype.DrawFood = function(ctx, cellWidth, cellHeight) {
    ctx.fillStyle = '#f44336'
    ctx.beginPath()
    ctx.rect(this.food.x * cellWidth, this.food.y * cellHeight, cellWidth, cellHeight)
    ctx.fill()
    ctx.stroke()
}

Snake.prototype.DrawInfo = function(infoBox) {
    let total = this.wall + this.eatSelf + this.noEat

    infoBox.innerText = `Длина змеи: ${this.snake.length} (максимальная: ${this.maxLength})\n`

    if (total == 0)
        return

    let ends = [
        `стена: ${this.wall} (${(this.wall / total * 100).toFixed(2)}%)`,
        `змея: ${this.eatSelf} (${(this.eatSelf / total * 100).toFixed(2)}%)`,
        `голод: ${this.noEat} (${(this.noEat / total * 100).toFixed(2)}%)`
    ]

    infoBox.innerText += `Конец: ${ends.join(', ')}`
}

Snake.prototype.Draw = function(ctx, infoBox) {
    if (this.snake == null)
        return null

    let width = ctx.canvas.width
    let height = ctx.canvas.height
    let cellWidth = width / (this.fieldWidth + 1)
    let cellHeight = height / (this.fieldHeight + 1)

    ctx.clearRect(0, 0, width, height)
    this.DrawCells(ctx, cellWidth, cellHeight)
    this.DrawSnake(ctx, cellWidth, cellHeight)
    this.DrawFood(ctx, cellWidth, cellHeight)
    this.DrawInfo(infoBox)
}
