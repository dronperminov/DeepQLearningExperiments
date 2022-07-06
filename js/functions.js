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
