function RandomUniform(a, b) {
    return a + Math.random() * (b - a)
}

function Max(values) {
    let maxValue = values[0]

    for (let value of values)
        if (value > maxValue)
            maxValue = value

    return maxValue
}

function Argmax(values) {
    let imax = 0

    for (let i = 1; i < values.length; i++)
        if (values[i] > values[imax])
            imax = i

    return imax
}

function Mean(values) {
    let mean = 0

    for (let value of values)
        mean += value

    return mean / values.length
}

function Std(values, mean) {
    let std = 0

    for (let value of values)
        std += (value - mean) * (value - mean)

    return Math.sqrt(std / values.length)
}

function Standartize(values) {
    let mean = Mean(values)
    let std = Std(values, mean)

    if (std < 1e-10)
        return

    for (let i = 0; i < values.length; i++)
        values[i] = (values[i] - mean) / std
}

function InitTensorMemory(size) {
    let tensor = []

    for (let i = 0; i < size[0]; i++) {
        tensor[i] = []

        for (let j = 0; j < size[1]; j++) {
            tensor[i][j] = []

            for (let k = 0; k < size[2]; k++)
                tensor[i][j][k] = 0
        }
    }

    return tensor
}
