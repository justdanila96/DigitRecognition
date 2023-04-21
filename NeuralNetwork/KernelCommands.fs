namespace NeuralNetwork

open ILGPU
open ILGPU.Runtime
open ILGPU.Algorithms

module KernelCommands =

    type Vector = ArrayView<float32>
    type Matrix = ArrayView2D<float32, Stride2D.DenseX>

    let FeedForward (index: Index1D) (weights: Matrix) (prevNeurons: Vector) (biases: Vector) (neurons: Vector) =
        let mutable sum = 0f

        for j = 0 to prevNeurons.IntLength - 1 do
            sum <- sum + prevNeurons[j] * weights[j, index]

        sum <- sum + biases[index]
        neurons[index] <- 1f / (1f + XMath.Exp(-sum))

    let CalcErrors (index: Index1D) (targets: Vector) (neurons: Vector) (errors: Vector) =
        errors[index] <- targets[index] - neurons[index]

    let CalcGradients (index: Index1D) (errors: Vector) (nextNeurons: Vector) (rate: float32) (gradients: Vector) =
        let deriv = nextNeurons[index] * (1f - nextNeurons[index])
        gradients[index] <- errors[index] * deriv * rate

    let CalcDeltas (index: Index2D) (gradients: Vector) (neurons: Vector) (deltas: Matrix) =
        deltas[index.X, index.Y] <- gradients[index.X] * neurons[index.Y]

    let CalcNextErrors (index: Index1D) (weights: Matrix) (errors: Vector) (nextErrors: Vector) =
        let mutable sum = 0f

        for j = 0 to errors.IntLength - 1 do
            sum <- sum + weights[index, j] * errors[j]

        nextErrors[index] <- sum

    let CalcNewWeights (index: Index2D) (deltas: Matrix) (weights: Matrix) =
        weights[index.X, index.Y] <- weights[index.X, index.Y] + deltas[index.Y, index.X]

    let CalcBiases (index: Index1D) (gradients: Vector) (biases: Vector) =
        biases[index] <- biases[index] + gradients[index]
