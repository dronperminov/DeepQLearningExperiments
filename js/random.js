function RandomUniform(a, b) {
    return a + Math.random() * (b - a)
}

function RandomSample(values, count) {
    let sample = []

    for (let i = 0; i < count; i++) {
        let index = Math.floor(Math.random() * values.length)
        sample.push(values[index])
    }

    return sample
}
