const SNAKE_TURN_LEFT = 1
const SNAKE_TURN_RIGHT = 2

const SNAKE_EAT_SELF = 'eat self'
const SNAKE_WALL = 'wall'
const SNAKE_EAT_FOOD = 'eat food'
const SNAKE_DEFAULT = 'default'

function Snake(canvas, fieldWidth = 29, fieldHeight = 19) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.fieldWidth = fieldWidth
    this.fieldHeight = fieldHeight

    this.snake = null
    this.food = null
    this.direction = null

    this.actionSpace = new DiscreteSpace(3)
    this.observationSpace = new UniformSpace(-1, 1, 25)
    this.maxLength = 0
}

Snake.prototype.InitSnake = function() {
    let snake = []

    let x0 = Math.floor(this.fieldWidth / 2)
    let y0 = Math.floor(this.fieldHeight / 2)

    snake.push({x: x0, y: y0})
    snake.push({x: x0, y: y0 + 1})
    snake.push({x: x0, y: y0 + 2})

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

    if (head.x + dx < 0 || head.y + dy < 0 || head.x + dx > this.fieldWidth || head.y + dy > this.fieldHeight)
        return SNAKE_WALL

    if (this.IsInsideSnake(head.x + dx, head.y + dy, 1))
        return SNAKE_EAT_SELF

    if (head.x + dx == this.food.x && head.y + dy == this.food.y) {
        this.snake.unshift({x: this.food.x, y: this.food.y})
        this.food = this.InitFood()
        return SNAKE_EAT_FOOD
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

Snake.prototype.DistanceToFood = function() {
    let dx = this.snake[0].x - this.food.x
    let dy = this.snake[0].y - this.food.y

    return Math.abs(dx) + Math.abs(dy)
}

Snake.prototype.DistanceToBody = function(dx, dy) {
    let x = this.snake[0].x + dx
    let y = this.snake[0].y + dy

    for (let i = 0; 0 <= x && x <= this.fieldWidth && 0 <= y && y <= this.fieldHeight; i++) {
        if (this.IsInsideSnake(x, y))
            return { dx: dx * (i + 1) / this.fieldWidth, dy: dy * (i + 1) / this.fieldHeight}

        x += dx
        y += dy
    }

    return { dx: dx, dy: dy }
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

    let body = this.DistanceToBody(this.direction.dx, this.direction.dy)
    let bodyLeft = this.DistanceToBody(this.direction.dy, -this.direction.dx)
    let bodyRight = this.DistanceToBody(-this.direction.dy, this.direction.dx)

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
        
        body.dx,
        body.dy,

        bodyLeft.dx,
        bodyLeft.dy,

        bodyRight.dx,
        bodyRight.dy,
    ]
}

Snake.prototype.Reset = function() {
    this.snake = this.InitSnake()
    this.food = this.InitFood()
    this.direction = {
        dx: 0,
        dy: -1
    }

    return this.StateToVector()
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

    let prevDst = this.DistanceToFood()
    let move = this.MoveSnake(this.direction.dx, this.direction.dy)
    let currDst = this.DistanceToFood()
    let done = move == SNAKE_WALL || move == SNAKE_EAT_SELF
    let reward = -1

    if (done) {
        reward = move == SNAKE_WALL ? -100 : -120
        this.maxLength = Math.max(this.snake.length, this.maxLength)
        console.log(move)
        console.log("snake length:", this.snake.length)
        console.log("max snake length:", this.maxLength)
    }
    else if (move == SNAKE_EAT_FOOD) {
        reward = 30
    }
    else if (currDst < prevDst) {
        reward = 1
    }

    return {
        state: this.StateToVector(),
        reward: reward,
        done: done
    }
}

Snake.prototype.Draw = function() {
    if (this.snake == null)
        return null

    let width = this.canvas.width
    let height = this.canvas.height
    let cellWidth = width / (this.fieldWidth + 1)
    let cellHeight = height / (this.fieldHeight + 1)

    this.ctx.clearRect(0, 0, width, height)

    this.ctx.strokeStyle = '#ccc'
    this.ctx.beginPath()

    for (let i = 0; i <= this.fieldHeight; i++)
        for (let j = 0; j <= this.fieldWidth; j++)
            this.ctx.rect(j * cellWidth, i * cellHeight, cellWidth, cellHeight)

    this.ctx.stroke()

    for (let p of this.snake) {
        this.ctx.fillStyle = '#4caf50'
        this.ctx.beginPath()
        this.ctx.rect(p.x * cellWidth, p.y * cellHeight, cellWidth, cellHeight)
        this.ctx.fill()
        this.ctx.stroke()
    }

    this.ctx.fillStyle = '#f44336'
    this.ctx.beginPath()
    this.ctx.rect(this.food.x * cellWidth, this.food.y * cellHeight, cellWidth, cellHeight)
    this.ctx.fill()
    this.ctx.stroke()
}